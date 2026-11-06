"""Model monitoring: input drift (PSI) and forecast-performance drift.

PSI compares each feature's distribution in a reference window (the 12 months the
final model trained on) against the most recent 8 weeks. Thresholds (industry
convention): < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate.
Performance drift: weekly store-week WAPE and bias with control limits from the
first half of the out-of-sample period; flags weeks outside mean +/- 3 sd.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sic import config
from sic.forecast.features import load_base

FEATURES = ["sales", "customers", "basket", "promo", "school_holiday", "competition_active",
            "promo2_active", "days_since_reopen", "lvl_recent"]


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if ref.nunique() <= 2:  # binary: compare proportions directly
        edges = None
        p = np.array([(ref == v).mean() for v in sorted(ref.unique())])
        q = np.array([(cur == v).mean() for v in sorted(ref.unique())])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        p = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
        q = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def status(v: float) -> str:
    return "stable" if v < 0.10 else "watch" if v < 0.25 else "investigate"


def run() -> dict:
    df = load_base()
    df = df[df.is_open == 1]
    end = df.date.max()
    cur = df[df.date > end - pd.Timedelta(weeks=8)]
    ref = df[(df.date >= "2014-05-01") & (df.date < "2015-05-01")]
    # same calendar months last year: separates true drift from seasonality
    ly = df[(df.date > end - pd.Timedelta(weeks=8) - pd.Timedelta(days=364))
            & (df.date <= end - pd.Timedelta(days=364))]
    feats = []
    for f in FEATURES:
        v = psi(ref[f], cur[f])
        v_ly = psi(ly[f], cur[f]) if len(ly) else float("nan")
        feats.append({"feature": f, "psi_vs_training_year": round(v, 4), "status": status(v),
                      "psi_vs_same_weeks_last_year": round(v_ly, 4),
                      "status_seasonal_adjusted": status(v_ly)})

    w = pd.read_parquet(config.path("outputs") / "store_weeks.parquet")
    wk = w.groupby("week_start").apply(lambda g: pd.Series({
        "wape": (g.sales - g.forecast).abs().sum() / g.sales.sum(),
        "bias": (g.forecast.sum() - g.sales.sum()) / g.sales.sum()}), include_groups=False)
    base = wk.iloc[: len(wk) // 2]
    alerts = []
    for m in ("wape", "bias"):
        mu, sd = base[m].mean(), base[m].std()
        out = wk[(wk[m] > mu + 3 * sd) | (wk[m] < mu - 3 * sd)]
        alerts += [{"metric": m, "week": str(i.date()), "value": round(float(v), 4),
                    "limits": [round(mu - 3 * sd, 4), round(mu + 3 * sd, 4)]}
                   for i, v in out[m].items()]
    # Run rule: a small but persistent bias matters even inside the 3-sd limits.
    run_len, run_sign, longest = 0, 0, None
    for i, b in wk.bias.items():
        sgn = int(np.sign(b)) if abs(b) >= 0.02 else 0
        run_len = run_len + 1 if sgn != 0 and sgn == run_sign else (1 if sgn != 0 else 0)
        run_sign = sgn
        if run_len >= 6:
            longest = {"metric": "bias", "rule": ">=6 consecutive weeks, same sign, |bias|>=2%",
                       "direction": "under-forecast" if sgn < 0 else "over-forecast",
                       "weeks": run_len, "through": str(i.date())}
    if longest:
        alerts.append(longest)
    recent = wk.tail(8)
    result = {
        "reference_window": "2014-05-01 to 2015-04-30 (final model's training year)",
        "current_window": f"8 weeks to {end.date()}",
        "features": feats,
        "performance": {"recent_8wk_wape": float(recent.wape.mean()),
                        "recent_8wk_bias": float(recent.bias.mean()),
                        "baseline_wape": float(base.wape.mean()),
                        "control_limit_alerts": alerts},
        "weekly": wk.reset_index().assign(week_start=lambda x: x.week_start.astype(str))
                    .round(4).to_dict(orient="records"),
    }
    (config.path("reports") / "m5_drift.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "weekly"}, indent=1))
