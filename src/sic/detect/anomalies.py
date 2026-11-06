"""M2: anomaly detection on out-of-sample forecasts.

1. Calibrate intervals (conformalized quantile regression, past-only):
   fold k's interval is widened by the 90th-pct conformity score observed on
   fold k-1's out-of-sample days. Fold 1 has no history -> kept uncalibrated.
2. Daily flag: actual outside the calibrated interval (direction +1 / -1).
3. Store-week anomaly: |actual/forecast - 1| >= min gap AND most open days
   flagged in the same direction.
4. Episodes: consecutive anomalous weeks for a store in the same direction.
5. Segment anomalies: store-type x week gaps that are extreme vs. that
   segment's own history (robust z-score).

Writes data/outputs/{forecasts_calibrated,store_weeks,episodes,segment_weeks}.parquet
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sic import config


def calibrate(fc: pd.DataFrame, level: float) -> tuple[pd.DataFrame, dict]:
    fc = fc.copy()
    m = (fc.is_open == 1) & (fc.sales > 0)
    ly, llo, lhi = (np.log1p(fc[c]) for c in ("sales", "lo", "hi"))
    fc["score"] = np.where(m, np.maximum(llo - ly, ly - lhi), np.nan)
    fc["lo_cal"], fc["hi_cal"] = fc.lo, fc.hi
    widen = {}
    for k in sorted(fc.fold.unique())[1:]:
        q = float(np.nanquantile(fc.loc[fc.fold == k - 1, "score"], level))
        widen[int(k)] = q
        idx = fc.fold == k
        fc.loc[idx, "lo_cal"] = np.expm1(np.log1p(fc.loc[idx, "lo"]) - q).clip(lower=0)
        fc.loc[idx, "hi_cal"] = np.expm1(np.log1p(fc.loc[idx, "hi"]) + q)
    cov = {}
    for k, g in fc[m].groupby("fold"):
        cov[int(k)] = {
            "raw": float(((g.sales >= g.lo) & (g.sales <= g.hi)).mean()),
            "calibrated": float(((g.sales >= g.lo_cal) & (g.sales <= g.hi_cal)).mean()),
        }
    return fc.drop(columns="score"), {"log_widening": widen, "coverage": cov}


def store_weeks(fc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    o = fc[fc.is_open == 1].copy()
    o["flag"] = np.select([o.sales > o.hi_cal, o.sales < o.lo_cal], [1, -1], 0)
    w = o.groupby(["store_id", "week_start", "store_type"]).agg(
        sales=("sales", "sum"), forecast=("p50", "sum"),
        customers=("customers", "sum"), forecast_customers=("cust_p50", "sum"),
        open_days=("sales", "size"), up_days=("flag", lambda s: (s == 1).sum()),
        down_days=("flag", lambda s: (s == -1).sum()), fold=("fold", "first"),
    ).reset_index()
    w["gap"] = w.sales - w.forecast
    w["gap_pct"] = w.sales / w.forecast - 1
    w["direction"] = np.sign(w.gap).astype(int)
    same_dir = np.where(w.direction > 0, w.up_days, w.down_days)
    w["share_days_outside"] = same_dir / w.open_days
    w["is_anomaly"] = (
        (w.open_days >= cfg["min_open_days"])
        & (w.gap_pct.abs() >= cfg["min_weekly_gap_pct"])
        & (w.share_days_outside >= cfg["min_share_days_outside"])
    )
    return w


def episodes(w: pd.DataFrame) -> pd.DataFrame:
    a = w[w.is_anomaly].sort_values(["store_id", "week_start"]).copy()
    new = (
        (a.store_id != a.store_id.shift())
        | (a.direction != a.direction.shift())
        | ((a.week_start - a.week_start.shift()).dt.days > 7)
    )
    a["episode_id"] = new.cumsum()
    e = a.groupby("episode_id").agg(
        store_id=("store_id", "first"), store_type=("store_type", "first"),
        direction=("direction", "first"), start=("week_start", "min"),
        end=("week_start", "max"), weeks=("week_start", "size"),
        sales=("sales", "sum"), forecast=("forecast", "sum"),
        customers=("customers", "sum"), forecast_customers=("forecast_customers", "sum"),
    ).reset_index()
    e["end"] = e["end"] + pd.Timedelta(days=6)
    e["gap"] = e.sales - e.forecast
    e["gap_pct"] = e.sales / e.forecast - 1
    e["severity"] = e.gap.abs()
    return e.sort_values("severity", ascending=False).reset_index(drop=True)


def segment_weeks(w: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
    s = w.groupby(["store_type", "week_start"]).agg(
        sales=("sales", "sum"), forecast=("forecast", "sum"), stores=("store_id", "nunique"),
    ).reset_index()
    s["gap_pct"] = s.sales / s.forecast - 1
    g = s.groupby("store_type").gap_pct
    med = g.transform("median")
    mad = g.transform(lambda x: (x - x.median()).abs().median()) * 1.4826
    s["robust_z"] = (s.gap_pct - med) / mad
    s["is_anomaly"] = (s.robust_z.abs() >= z_threshold) & (s.stores >= 10)
    return s


def run() -> dict:
    cfg_f, cfg_d = config.load()["forecast"], config.load()["detect"]
    lo_q, hi_q = cfg_f["interval"]
    out = config.path("outputs")
    fc = pd.read_parquet(out / "forecasts.parquet")
    fc, cal = calibrate(fc, level=hi_q - lo_q)
    w = store_weeks(fc, cfg_d)
    e = episodes(w)
    s = segment_weeks(w)
    fc.to_parquet(out / "forecasts_calibrated.parquet", index=False)
    w.to_parquet(out / "store_weeks.parquet", index=False)
    e.to_parquet(out / "episodes.parquet", index=False)
    s.to_parquet(out / "segment_weeks.parquet", index=False)

    summary = {
        "calibration": cal,
        "store_weeks": len(w),
        "anomalous_store_weeks": int(w.is_anomaly.sum()),
        "anomaly_rate": float(w.is_anomaly.mean()),
        "episodes": len(e),
        "episodes_up": int((e.direction > 0).sum()),
        "episodes_down": int((e.direction < 0).sum()),
        "segment_anomalies": s[s.is_anomaly][["store_type", "week_start", "gap_pct", "robust_z"]]
        .assign(week_start=lambda x: x.week_start.astype(str)).round(3).to_dict(orient="records"),
        "top_episodes": e.head(10)[["store_id", "store_type", "direction", "start", "end",
                                    "weeks", "gap", "gap_pct"]]
        .assign(start=lambda x: x.start.astype(str), end=lambda x: x.end.astype(str))
        .round(3).to_dict(orient="records"),
    }
    (config.path("reports") / "m2_detection_summary.json").write_text(
        json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
