"""Feature selection for the daily sales forecaster.

Selection data is kept apart from evaluation data. Every step below runs on backtest folds
1-3 (config forecast.selection.windows): together they cover Christmas and Easter, and each
fold's training data contains the previous year's. Fold 4 (May-Jul 2015) is never looked at
during selection and is used once at the end to compare full vs selected features.
(A first version selected on windows before fold 1; its training data had no prior
Christmas/Easter, so seasonal features showed zero importance because they were untestable.)

Methods
  1. Redundancy filter    Spearman |rho| between features on training rows, hierarchical
                          clustering (average linkage, distance 1-|rho|). Clusters of
                          near-duplicates are reported; near-constant features are flagged.
  2. Permutation          out-of-sample WAPE increase when a feature (or a whole redundancy
     importance           cluster, permuted jointly) is shuffled in the validation window.
                          Grouped permutation avoids the classic trap where two correlated
                          features each look unimportant because the other covers for it.
                          (Impurity / split-gain importance is not used: biased toward
                          high-cardinality features and measured in-sample.)
  3. Group ablation       drop a domain group (calendar, promo sequence, ...), retrain, measure
                          the WAPE change: what each block of features is worth.
  4. Backward elimination wrapper method. Candidates = features whose permutation importance
                          is below candidate_max_importance, weakest first. For each, retrain
                          without it; keep it dropped if mean WAPE rises <= drop_tolerance and
                          no window rises > drop_tolerance_per_window. Retraining after every
                          drop means a correlated partner that becomes essential is kept.
                          Untestable features (constant in every selection window) are kept.
                          The cumulative change vs the full model is also capped at
                          drop_tolerance, so many tiny drops cannot add up.

Delta-WAPE uncertainty: store-level bootstrap (stores x windows resampled) of paired errors.
Prior art (methods only; no code copied):
  - scikit-learn example "Permutation Importance with Multicollinear or Correlated Features"
    (BSD-3): Spearman + hierarchical clustering of features. We use average linkage on
    1-|rho| (Ward assumes Euclidean distances) and add joint (grouped) permutation.
  - Rossmann Kaggle 1st place write-up (G. Jacobusse): feature selection on a separate
    time-based holdout to avoid overfitting -> our pre-backtest selection windows.
  - scikit-learn SequentialFeatureSelector / skforecast feature-selection guide (BSD-3):
    backward selection scored on time-ordered validation -> our backward elimination, plus a
    parsimony tolerance and a cumulative cap.
  - Boruta (scikit-learn-contrib/boruta_py, BSD-3) is common in public Rossmann repos; not
    used here because it relies on in-sample impurity importance.

Writes reports/m2_feature_selection.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sic import config
from sic.forecast import features as F
from sic.forecast.backtest import gbm

GROUPS = {
    "calendar": ["dow", "day", "month", "weekofyear", "dayofyear", "year"],
    "promo": ["promo"],
    "holidays": ["state_holiday", "school_holiday"],
    "store_attributes": ["store_type", "assortment"],
    "competition": ["competition_active", "months_since_competition_open",
                    "competition_distance"],
    "promo2": ["promo2_active", "promo2_renewal_month"],
    "closures": ["closed_yesterday", "closed_tomorrow", "days_since_reopen"],
    "holiday_proximity (v2)": ["days_to_closure", "days_since_closure", "easter_offset",
                               "days_to_christmas"],
    "promo_sequence (v2)": ["promo_day_index", "promo_last_week", "promo_next_week"],
    "recent_level (v2)": ["lvl_recent", "days_since_anchor", "lvl_vs_longrun"],
    "store_encodings": ["te_store", "te_store_dow", "te_store_promo", "te_store_cust"],
}
assert sorted(f for g in GROUPS.values() for f in g) == sorted(F.FEATURES), "GROUPS must cover FEATURES"


def _cfg() -> dict:
    return config.load()["forecast"]["selection"]


# ------------------------------------------------------------------ data
def build_windows() -> list[dict]:
    base = F.load_base()
    out = []
    for w in _cfg()["windows"]:
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        train = base[base.date < start]
        test = base[(base.date >= start) & (base.date <= end)]
        test = test[(test.is_open == 1) & (test.sales > 0)]
        fit_rows = F.target_encode(train, train[(train.is_open == 1) & (train.sales > 0)])
        test = F.target_encode(train, test)
        out.append({"name": f"{w['start']}..{w['end']}",
                    "X": F.to_matrix(fit_rows, F.FEATURES), "y": np.log1p(fit_rows.sales.values),
                    "Xt": F.to_matrix(test, F.FEATURES), "sales": test.sales.values,
                    "store": test.store_id.values})
    return out


def wape(pred: np.ndarray, sales: np.ndarray) -> float:
    return float(np.abs(pred - sales).sum() / sales.sum())


def fit_predict(w: dict, feats: list[str]) -> tuple[object, np.ndarray]:
    m = gbm().fit(w["X"][feats], w["y"])
    return m, np.expm1(m.predict(w["Xt"][feats]))


def evaluate(windows: list[dict], feats: list[str]) -> dict:
    preds = [fit_predict(w, feats)[1] for w in windows]
    per = [wape(p, w["sales"]) for p, w in zip(preds, windows)]
    return {"wape": per, "mean": float(np.mean(per)), "preds": preds}


def bootstrap_delta(windows: list[dict], pa: list, pb: list, n: int, seed: int = 7) -> list:
    """95% CI of mean-over-windows WAPE(b) - WAPE(a), resampling stores within each window."""
    rng = np.random.default_rng(seed)
    per_window = []
    for w, a, b in zip(windows, pa, pb):
        df = pd.DataFrame({"s": w["store"], "ea": np.abs(a - w["sales"]),
                           "eb": np.abs(b - w["sales"]), "y": w["sales"]})
        g = df.groupby("s")[["ea", "eb", "y"]].sum().values
        idx = rng.integers(0, len(g), size=(n, len(g)))
        t = g[idx].sum(axis=1)                                    # n x 3
        per_window.append(t[:, 1] / t[:, 2] - t[:, 0] / t[:, 2])
    d = np.mean(per_window, axis=0)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ------------------------------------------------------------------ 1. redundancy
def redundancy(X: pd.DataFrame, threshold: float) -> dict:
    num = X.copy()
    for c in F.CATEGORICAL:
        num[c] = num[c].cat.codes
    num = num.sample(min(len(num), 200_000), random_state=7)
    rho = num.rank().corr().abs().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(rho, 1.0)
    dist = squareform(1 - rho, checks=False)
    labels = fcluster(linkage(dist, "average"), t=1 - threshold, criterion="distance")
    clusters = {}
    for f, lab in zip(num.columns, labels):
        clusters.setdefault(int(lab), []).append(f)
    multi = [sorted(v) for v in clusters.values() if len(v) > 1]
    pairs = []
    cols = list(num.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if rho[i, j] >= threshold:
                pairs.append({"a": cols[i], "b": cols[j], "spearman_abs": round(float(rho[i, j]), 3)})
    top_share = {c: float(num[c].value_counts(normalize=True).iloc[0]) for c in cols}
    near_constant = {c: round(s, 4) for c, s in top_share.items() if s > 0.99}
    return {"threshold": threshold, "clusters": multi,
            "pairs": sorted(pairs, key=lambda p: -p["spearman_abs"]),
            "near_constant": near_constant}


# ------------------------------------------------------------------ 2. permutation importance
def permutation_importance(windows: list[dict], units: dict[str, list[str]], repeats: int) -> dict:
    rng = np.random.default_rng(7)
    res = {u: [] for u in units}
    for w in windows:
        model, pred = fit_predict(w, F.FEATURES)
        ref = wape(pred, w["sales"])
        for u, cols in units.items():
            deltas = []
            for _ in range(repeats):
                Xp = w["Xt"].copy()
                perm = rng.permutation(len(Xp))
                for c in cols:                       # same permutation -> joint shuffle
                    Xp[c] = Xp[c].values[perm]
                    if c in F.CATEGORICAL:
                        Xp[c] = Xp[c].astype(w["Xt"][c].dtype)
                deltas.append(wape(np.expm1(model.predict(Xp)), w["sales"]) - ref)
            res[u].append(deltas)
    return {u: {"mean": float(np.mean(v)), "std": float(np.std(np.ravel(v))),
                "by_window": [float(np.mean(x)) for x in v], "features": units[u]}
            for u, v in res.items()}


# ------------------------------------------------------------------ held-out check
def holdout(selected: list[str]) -> dict:
    """Full vs selected features on the held-out fold (never used in selection)."""
    cfg = _cfg()
    fold = config.load()["forecast"]["folds"][cfg["holdout_fold"] - 1]
    base = F.load_base()
    start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
    train = base[base.date < start]
    test = base[(base.date >= start) & (base.date <= end)]
    test = test[(test.is_open == 1) & (test.sales > 0)]
    fit_rows = F.target_encode(train, train[(train.is_open == 1) & (train.sales > 0)])
    test = F.target_encode(train, test)
    w = {"X": F.to_matrix(fit_rows, F.FEATURES), "y": np.log1p(fit_rows.sales.values),
         "Xt": F.to_matrix(test, F.FEATURES), "sales": test.sales.values,
         "store": test.store_id.values}
    out, preds = {}, {}
    for name, feats in (("full", list(F.FEATURES)), ("selected", selected)):
        p = fit_predict(w, feats)[1]
        preds[name] = p
        wk = pd.DataFrame({"s": w["store"], "wk": test.week_start.values, "p": p,
                           "y": w["sales"]}).groupby(["s", "wk"])[["p", "y"]].sum()
        out[name] = {"n_features": len(feats), "daily_wape": wape(p, w["sales"]),
                     "store_week_wape": float(np.abs(wk.p - wk.y).sum() / wk.y.sum())}
    out["delta_daily_wape_ci95"] = bootstrap_delta([w], [preds["full"]], [preds["selected"]],
                                                   cfg["bootstrap"])
    out["fold"] = fold
    out["reading"] = ("Selected features are adopted only if held-out error is not worse "
                      "beyond the bootstrap CI.")
    (config.path("reports") / "m2_feature_selection_holdout.json").write_text(
        json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ main
def _checkpoint_path():
    return config.path("outputs") / "select_checkpoint.json"


def _load_checkpoint(key: str) -> dict:
    """Resume state from an interrupted run, if it was made with the same settings."""
    p = _checkpoint_path()
    if p.exists():
        ck = json.loads(p.read_text())
        if ck.get("key") == key:
            return ck
    return {"key": key}


def _save_checkpoint(ck: dict) -> None:
    tmp = _checkpoint_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(ck))
    tmp.replace(_checkpoint_path())


def run() -> dict:
    """Long (~1-2 h). Thread count is capped (selection.n_threads) so the run does not
    oversubscribe the CPU, and progress is checkpointed after every model fit group, so an
    interrupted run resumes where it stopped (same settings only)."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=_cfg().get("n_threads", 8)):
        return _run()


def _run() -> dict:
    cfg = _cfg()
    t0 = time.time()
    key = json.dumps({"cfg": {k: v for k, v in cfg.items() if k != "n_threads"},
                      "features": F.FEATURES,
                      "gbm": config.load()["forecast"]["gbm"]}, sort_keys=True)
    ck = _load_checkpoint(key)
    windows = build_windows()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    log(f"windows ready: {[(w['name'], len(w['X']), len(w['Xt'])) for w in windows]}"
        + (" (resuming from checkpoint)" if len(ck) > 1 else ""))

    red = redundancy(windows[-1]["X"], cfg["redundancy_spearman"])
    log(f"redundancy clusters: {red['clusters']}")

    full = evaluate(windows, F.FEATURES)
    log(f"full model WAPE {full['wape']}")

    singles = {f: [f] for f in F.FEATURES}
    clusters = {"+".join(c): c for c in red["clusters"]}
    if "pi" not in ck:
        ck["pi"] = permutation_importance(windows, {**singles, **clusters},
                                          cfg["permutation_repeats"])
        _save_checkpoint(ck)
    pi = ck["pi"]
    log("permutation importance done")

    ablation = ck.setdefault("ablation", {})
    for g, cols in GROUPS.items():
        if g in ablation:
            continue
        r = evaluate(windows, [f for f in F.FEATURES if f not in cols])
        ablation[g] = {"features": cols, "delta_wape": r["mean"] - full["mean"],
                       "delta_by_window": [b - a for a, b in zip(full["wape"], r["wape"])],
                       "ci95": bootstrap_delta(windows, full["preds"], r["preds"], cfg["bootstrap"])}
        _save_checkpoint(ck)
        log(f"ablation {g}: {ablation[g]['delta_wape']:+.4f}")

    # backward elimination
    current, cur_eval = list(F.FEATURES), full
    untestable = [f for f in F.FEATURES if not any(
        w["X"][f].nunique() > 1 and w["Xt"][f].nunique() > 1 for w in windows)]
    candidates = sorted((f for f in F.FEATURES if pi[f]["mean"] < cfg["candidate_max_importance"]
                         and f not in untestable), key=lambda f: pi[f]["mean"])
    log(f"untestable (kept): {untestable}; candidates: {candidates}")
    steps = ck.setdefault("steps", [])
    done = {st["feature"] for st in steps}
    dropped_so_far = [st["feature"] for st in steps if st["dropped"]]
    if dropped_so_far:                     # resume: rebuild the current set, refit it once
        current = [c for c in F.FEATURES if c not in dropped_so_far]
        cur_eval = evaluate(windows, current)
    for f in candidates:
        if f in done:
            continue
        trial = [c for c in current if c != f]
        r = evaluate(windows, trial)
        d = r["mean"] - cur_eval["mean"]
        d_win = [b - a for a, b in zip(cur_eval["wape"], r["wape"])]
        cum = r["mean"] - full["mean"]          # guard against many small drops adding up
        ok = (d <= cfg["drop_tolerance"] and max(d_win) <= cfg["drop_tolerance_per_window"]
              and cum <= cfg["drop_tolerance"])
        steps.append({"feature": f, "perm_importance": pi[f]["mean"], "delta_wape": d,
                      "cumulative_delta_vs_full": cum,
                      "delta_by_window": d_win, "dropped": bool(ok),
                      "ci95": bootstrap_delta(windows, cur_eval["preds"], r["preds"],
                                              cfg["bootstrap"])})
        _save_checkpoint(ck)
        log(f"try drop {f}: {d:+.5f} {d_win} -> {'DROP' if ok else 'keep'}")
        if ok:
            current, cur_eval = trial, r

    dropped = [s["feature"] for s in steps if s["dropped"]]
    result = {
        "selection_windows": [w["name"] for w in windows],
        "note": ("Selection uses backtest folds 1-3 only; fold "
                 f"{cfg['holdout_fold']} is the held-out comparison."),
        "untestable_kept": untestable,
        "settings": cfg,
        "n_features_full": len(F.FEATURES),
        "full_wape": {"by_window": full["wape"], "mean": full["mean"]},
        "redundancy": red,
        "permutation_importance": dict(sorted(pi.items(), key=lambda kv: -kv[1]["mean"])),
        "group_ablation": dict(sorted(ablation.items(), key=lambda kv: -kv[1]["delta_wape"])),
        "elimination": sorted(steps, key=lambda st: candidates.index(st["feature"])),
        "selected_features": current, "dropped_features": dropped,
        "selected_wape": {"by_window": cur_eval["wape"], "mean": cur_eval["mean"]},
        "seconds": round(time.time() - t0),
    }
    (config.path("reports") / "m2_feature_selection.json").write_text(json.dumps(result, indent=1))
    log(f"selected {len(current)}/{len(F.FEATURES)}; dropped {dropped}")
    ho = holdout(current)
    log(f"held-out fold: {json.dumps({k: ho[k] for k in ('full', 'selected')})} "
        f"delta CI {ho['delta_daily_wape_ci95']}")
    return result


def holdout_from_backtests(old_fc_path: str) -> dict:
    """Held-out comparison from two production backtests: the all-features forecasts
    (backed up before the switch) vs the current forecasts.parquet (selected features).
    Fold `holdout_fold` was never used in selection; folds 1-3 are shown for reference only
    (they were used to choose the features, so they are optimistic)."""
    cfg = _cfg()
    old = pd.read_parquet(old_fc_path)
    new = pd.read_parquet(config.path("outputs") / "forecasts.parquet")
    key = ["store_id", "date"]
    m = old[key + ["fold", "sales", "is_open", "week_start", "p50"]].merge(
        new[key + ["p50"]], on=key, suffixes=("_full", "_sel"))
    m = m[(m.is_open == 1) & (m.sales > 0)]
    out = {"holdout_fold": cfg["holdout_fold"], "by_fold": {}}
    for fold, g in m.groupby("fold"):
        wk = g.groupby(["store_id", "week_start"])[["sales", "p50_full", "p50_sel"]].sum()
        r = {}
        for name, col in (("full", "p50_full"), ("selected", "p50_sel")):
            r[name] = {"daily_wape": wape(g[col].values, g.sales.values),
                       "store_week_wape": float(np.abs(wk[col] - wk.sales).sum()
                                                / wk.sales.sum())}
        w = {"store": g.store_id.values, "sales": g.sales.values}
        r["delta_daily_wape_ci95"] = bootstrap_delta(
            [w], [g.p50_full.values], [g.p50_sel.values], cfg["bootstrap"])
        r["used_in_selection"] = int(fold) != cfg["holdout_fold"]
        out["by_fold"][int(fold)] = r
    ho = out["by_fold"][cfg["holdout_fold"]]
    out.update({"full": ho["full"], "selected": ho["selected"],
                "delta_daily_wape_ci95": ho["delta_daily_wape_ci95"],
                "n_features": {"full": len(F.FEATURES), "selected": len(F.active_features())},
                "adopt": ho["delta_daily_wape_ci95"][0] <= 0,
                "rule": "adopt if the held-out delta CI includes or is below 0 (not worse)"})
    (config.path("reports") / "m2_feature_selection_holdout.json").write_text(
        json.dumps(out, indent=1))
    return out


def report_from_log(log_path: str) -> dict:
    """Rebuild reports/m2_feature_selection.json from the log of a run that was stopped
    before it finished (the first folds-1-3 run predates checkpointing). Only what the log
    printed is kept: full-model WAPE, group-ablation means, and every elimination decision.
    Permutation-importance values and bootstrap CIs were not logged and are left out."""
    import ast
    import re

    cfg = _cfg()
    from pathlib import Path

    text = Path(log_path).read_text()
    full = ast.literal_eval(re.search(r"full model WAPE (\[.*?\])", text).group(1))
    clusters = ast.literal_eval(re.search(r"redundancy clusters: (\[\[.*?\]\])", text).group(1))
    ablation = {g: {"features": GROUPS[g], "delta_wape": float(v), "ci95": None}
                for g, v in re.findall(r"ablation (.+?): ([+-][\d.]+)", text)}
    cands = ast.literal_eval(re.search(r"candidates: (\[.*?\])", text).group(1))
    steps, current = [], list(F.FEATURES)
    for f, d, win, dec in re.findall(r"try drop (\S+): ([+-][\d.]+) (\[.*?\]) -> (DROP|keep)", text):
        steps.append({"feature": f, "delta_wape": float(d), "delta_by_window":
                      ast.literal_eval(win), "dropped": dec == "DROP", "ci95": None})
        if dec == "DROP":
            current.remove(f)
    tried = {st["feature"] for st in steps}
    result = {
        "partial": True,
        "selection_windows": [f"{w['start']}..{w['end']}" for w in cfg["windows"]],
        "note": (f"Run stopped after {len(steps)}/{len(cands)} elimination candidates (CPU "
                 "oversubscription made it ~10x slower); untried candidates are kept. "
                 "Permutation-importance values and bootstrap CIs were not logged."),
        "untried_kept": [c for c in cands if c not in tried],
        "settings": cfg, "n_features_full": len(F.FEATURES),
        "full_wape": {"by_window": full, "mean": float(np.mean(full))},
        "redundancy": {"threshold": cfg["redundancy_spearman"], "clusters": clusters,
                       "near_constant": {}},
        "permutation_importance": {},
        "candidates_in_order": cands,
        "group_ablation": dict(sorted(ablation.items(), key=lambda kv: -kv[1]["delta_wape"])),
        "elimination": steps,
        "selected_features": current,
        "dropped_features": [st["feature"] for st in steps if st["dropped"]],
    }
    (config.path("reports") / "m2_feature_selection.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--holdout-from":
        print(json.dumps(holdout_from_backtests(sys.argv[2]), indent=1))
    elif len(sys.argv) > 2 and sys.argv[1] == "--from-log":
        r = report_from_log(sys.argv[2])
        print(json.dumps({k: r[k] for k in ("dropped_features", "untried_kept", "note")}, indent=1))
    else:
        run()
