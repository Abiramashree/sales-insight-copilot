"""Initiative planner: turn a goal into a plan whose every number is computed here (no LLM).

plan(target_pct, store_type | store_ids | all, ...) returns:
  baseline      the scope's weekly sales (mean of the last 8 complete weeks) and the weekly €
                the goal requires
  plausibility  measured upside available in scope (risk-adjusted opportunity scores, see
                src/sic/opportunity) / required €: >= 1.5 Plausible, >= 0.75 Stretch, else
                Unlikely; plus the scope's normal week-to-week swing vs forecast
  stores        the fewest top-upside stores whose upside covers the goal, each with
                the € and % it must add (goal split by upside share)
  hypotheses    if/then/because statements, each tested on the selected stores' history
                (src/sic/hypothesis): promo effect, and whether it works via customers or basket
  alternatives  levers sized in €/week with a range: extra promo week (what-if Monte Carlo),
                Promo2 enrollment (what-if, low confidence), basket to peer level
                (observational; range = capture rate 10%-40% around the 25% assumption),
                competitor defense (DiD-based, if any recent competitor)
  experiment    A/B sizing for the lift the goal needs from the selected stores, a
                reproducible test/control list, decision rule and guardrail
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sic import config
from sic import experiment as X
from sic import hypothesis as H


def _scope(opp: pd.DataFrame, store_type: str | None, store_ids: list[int] | None) -> pd.DataFrame:
    if store_ids:
        s = opp[opp.store_id.isin([int(i) for i in store_ids])]
        missing = sorted({int(i) for i in store_ids} - set(s.store_id))
        if missing:
            raise ValueError(f"Unknown stores: {missing[:10]} (valid 1-1115).")
        return s
    if store_type:
        s = opp[opp.store_type == store_type.lower()]
        if s.empty:
            raise ValueError(f"No stores of type {store_type!r} (types: a, b, c, d).")
        return s
    return opp


def _baseline(ids: list[int], weeks: int = 8) -> pd.Series:
    w = X.weekly()
    recent = sorted(w.week_start.unique())[-weeks:]
    return w[w.week_start.isin(recent) & w.store_id.isin(ids)].groupby("store_id").sales.mean()


def _verdict(coverage: float) -> str:
    return "Plausible" if coverage >= 1.5 else "Stretch" if coverage >= 0.75 else "Unlikely"


def plan(target_pct: float, store_type: str | None = None, store_ids: list[int] | None = None,
         test_weeks: int = 8, seed: int = 7) -> dict:
    if not 0 < target_pct < 1:
        raise ValueError("target_pct is a fraction, e.g. 0.05 for 5%.")
    out_dir = config.path("outputs")
    opp = pd.read_parquet(out_dir / "opportunity_scores.parquet")
    scope = _scope(opp, store_type, store_ids)
    base = _baseline(scope.store_id.tolist())
    scope = scope[scope.store_id.isin(base.index)].copy()
    scope["baseline"] = scope.store_id.map(base)
    baseline_eur = float(scope.baseline.sum())
    need_eur = baseline_eur * target_pct

    # plausibility: measured upside vs required, and normal swing vs forecast
    upside = float(scope.score_eur.clip(lower=0).sum())
    coverage = upside / need_eur if need_eur else float("inf")
    sw = pd.read_parquet(out_dir / "store_weeks.parquet")
    sw = sw[sw.store_id.isin(scope.store_id) & (sw.open_days >= 5)]
    wk = sw.groupby("week_start")[["sales", "forecast"]].sum()
    swing = float((wk.sales / wk.forecast - 1).std(ddof=1))

    # stores: fewest top-upside stores that cover the goal
    ranked = scope[scope.score_eur > 0].sort_values("score_eur", ascending=False)
    cum = ranked.score_eur.cumsum()
    k = int(min((cum < need_eur).sum() + 1, len(ranked)))
    sel = ranked.head(k).copy()
    covered = float(min(sel.score_eur.sum(), need_eur))
    sel["add_eur"] = sel.score_eur / sel.score_eur.sum() * covered
    sel["add_pct"] = sel.add_eur / sel.baseline
    ids = sel.store_id.astype(int).tolist()
    stores = [{"store_id": int(r.store_id), "store_type": r.store_type,
               "weekly_sales": round(float(r.baseline)), "add_eur_per_week": round(float(r.add_eur)),
               "add_pct": round(float(r.add_pct), 4), "main_lever": r.primary_lever}
              for r in sel.itertuples()]

    # hypotheses, tested on the selected stores' history
    hyps = []
    if len(ids) >= 3:
        promo = H.test("sales", "promo", store_ids=ids)
        cust = H.test("customers", "promo", store_ids=ids)
        bask = H.test("basket", "promo", store_ids=ids)
        hyps = [
            {"id": "H1", "if": "the selected stores run one extra promo week per quarter",
             "then": "their weekly sales rise", "because":
             "promo weeks lift these stores' sales vs their own non-promo weekdays",
             "test": "within-store, weekday-matched promo vs no promo (sales)",
             **{k: promo[k] for k in ("effect_pct", "ci_pct", "p_value", "n_a")}},
            {"id": "H2", "if": "the uplift comes from more customer visits",
             "then": "customer counts rise more than basket size",
             "because": "promos mainly bring people in (traffic), not bigger baskets",
             "test": "within-store promo vs no promo: customers and basket",
             "customers_effect_pct": cust["effect_pct"], "customers_ci_pct": cust["ci_pct"],
             "basket_effect_pct": bask["effect_pct"], "basket_ci_pct": bask["ci_pct"],
             # supported only if the intervals don't overlap; overlapping = inconclusive
             "result": ("supported" if cust["ci_pct"][0] > bask["ci_pct"][1] else
                        "contradicted" if bask["ci_pct"][0] > cust["ci_pct"][1] else
                        "inconclusive (confidence intervals overlap)")},
        ]

    # alternatives, sized in €/week with a range
    from sic.whatif.engine import ScenarioError
    from sic.whatif.engine import run as sim

    alts = []
    if ids:
        r = sim(scenario="extra_promo_week", store_ids=ids)
        q = 13       # one extra promo week per quarter -> weekly equivalent
        alts.append({"option": f"One extra promo week per quarter in the {len(ids)} stores",
                     "eur_per_week": r["total_eur"]["mean"] / q,
                     "range": [r["total_eur"]["p5"] / q, r["total_eur"]["p95"] / q],
                     "confidence": r["confidence"], "basis": r["method"]})
    cap = config.load()["opportunity"]["peer_capture_rate"]
    peer = float(scope.eur_peer.clip(lower=0).sum())
    if peer > 0:
        alts.append({"option": "Lift basket toward look-alike stores (merchandising, upsell)",
                     "eur_per_week": peer, "range": [peer * 0.10 / cap, peer * 0.40 / cap],
                     "confidence": "low",
                     "basis": f"observational basket gap vs peers; range = capture 10%-40% "
                              f"(assumed {cap:.0%})"})
    not_enrolled = scope[~scope.store_id.isin(
        pd.read_parquet(out_dir / "promo_lift_by_store.parquet").query("promo2_enrolled")
        .store_id)].store_id.astype(int).tolist()
    if not_enrolled:
        try:
            r = sim(scenario="promo2_enroll", store_ids=not_enrolled, weeks=13)
            alts.append({"option": f"Enroll {len(not_enrolled)} stores in Promo2",
                         "eur_per_week": r["per_week_eur"]["mean"],
                         "range": [r["per_week_eur"]["p5"], r["per_week_eur"]["p95"]],
                         "confidence": r["confidence"], "basis": r["method"]})
        except ScenarioError:
            pass
    comp = float(scope.eur_competitor.sum())
    if comp > 0:
        alts.append({"option": "Defend stores with a recent nearby competitor",
                     "eur_per_week": comp, "range": [comp * 0.5, comp * 1.5],
                     "confidence": "medium",
                     "basis": "DiD competitor effect x protectable share (assumption-driven)"})
    alts.sort(key=lambda a: -a["eur_per_week"])

    # experiment: size for the lift the goal needs from the selected stores
    sel_base = float(sel.baseline.sum()) if len(sel) else 0.0
    needed_lift = covered / sel_base if sel_base else target_pct
    exp = {"note": "Fewer than 4 stores selected: too few to split into test and control. "
                   "Compare them before/after against look-alike stores instead."}
    if len(ids) >= 4:
        size = H.plan(weeks=int(np.clip(test_weeks, 2, 12)), stores_per_arm=len(ids) // 2,
                      target_lift=float(np.clip(needed_lift, 0.005, 0.5)))
        groups = X.assign(ids, seed=seed)
        exp = {**{k: size[k] for k in ("weeks", "stores_per_arm", "target_lift", "mde_cuped",
                                       "mde_raw", "stores_needed_cuped", "stores_needed_raw")},
               "powered": bool(size["stores_needed_cuped"] <= len(ids) // 2),
               "seed": seed,
               "groups": groups.to_dict(orient="records"),
               "primary_metric": "weekly sales per store (CUPED-adjusted)",
               "guardrail": "basket size must not fall more than 2%",
               "decision_rule": "roll out if the 95% CI is above 0 and the lift is at least "
                                "half the target lift"}

    return {"goal": {"target_pct": target_pct, "store_type": store_type,
                     "store_ids": store_ids, "scope_stores": len(scope)},
            "baseline": {"weekly_sales_eur": baseline_eur, "required_eur_per_week": need_eur},
            "plausibility": {"verdict": _verdict(coverage), "upside_eur_per_week": upside,
                             "coverage": coverage, "weekly_swing_vs_forecast": swing,
                             "rule": ">= 1.5x required: Plausible; >= 0.75x: Stretch; "
                                     "else Unlikely (upside = risk-adjusted opportunity "
                                     "scores)"},
            "stores": stores, "covered_eur_per_week": covered,
            "gap_eur_per_week": max(need_eur - covered, 0.0),
            "hypotheses": hyps, "alternatives": alts, "experiment": exp}
