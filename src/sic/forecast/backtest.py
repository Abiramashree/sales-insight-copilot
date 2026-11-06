"""M2: rolling-origin backtest of daily store-sales forecasters.

Models
  baseline  store x day-of-week x promo mean over the 13 weeks before the fold
  gbm       global gradient-boosted trees on log sales (sklearn HistGradientBoosting)
  gbm_q*    same features, quantile loss -> prediction interval
  cust      same model for customer counts (feeds traffic-vs-basket RCA)

Closed days are forecast as 0 (the opening schedule is known in advance).
Writes data/outputs/forecasts.parquet and reports/m2_backtest_metrics.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sic import config
from sic.forecast import features as F


def gbm(loss: str = "squared_error", quantile: float | None = None):
    params = dict(config.load()["forecast"]["gbm"])
    return HistGradientBoostingRegressor(
        loss=loss, quantile=quantile, categorical_features="from_dtype",
        early_stopping=False, random_state=7, **params)


def baseline(train: pd.DataFrame, test: pd.DataFrame, start: pd.Timestamp) -> np.ndarray:
    recent = train[(train.date >= start - pd.Timedelta(weeks=13)) & (train.is_open == 1)]
    k3 = recent.groupby(["store_id", "dow", "promo"]).sales.mean().rename("b3")
    k2 = train[train.is_open == 1].groupby(["store_id", "dow"]).sales.mean().rename("b2")
    out = test[["store_id", "dow", "promo"]].join(k3, on=["store_id", "dow", "promo"])
    out = out.join(k2, on=["store_id", "dow"])
    return out.b3.fillna(out.b2).fillna(0).values


def metrics(df: pd.DataFrame, col: str) -> dict:
    m = df[(df.is_open == 1) & (df.sales > 0)]
    err = m[col] - m.sales
    return {
        "rmspe": float(np.sqrt(np.mean((err / m.sales) ** 2))),
        "wape": float(np.abs(err).sum() / m.sales.sum()),
        "bias": float(err.sum() / m.sales.sum()),
    }


def run() -> dict:
    cfg = config.load()["forecast"]
    lo_q, hi_q = cfg["interval"]
    base = F.load_base()
    preds, fold_metrics = [], []

    for i, fold in enumerate(cfg["folds"], start=1):
        t0 = time.time()
        start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        train = base[base.date < start]
        test = base[(base.date >= start) & (base.date <= end)].copy()

        fit_rows = F.target_encode(train, train[(train.is_open == 1) & (train.sales > 0)])
        test = F.target_encode(train, test)
        X, Xt = F.to_matrix(fit_rows), F.to_matrix(test)
        y, yc = np.log1p(fit_rows.sales), np.log1p(fit_rows.customers)

        open_ = test.is_open.values == 1
        test["fold"] = i
        test["baseline"] = np.where(open_, baseline(train, test, start), 0)
        test["p50"] = np.where(open_, np.expm1(gbm().fit(X, y).predict(Xt)), 0)
        test["lo"] = np.where(open_, np.expm1(gbm("quantile", lo_q).fit(X, y).predict(Xt)), 0)
        test["hi"] = np.where(open_, np.expm1(gbm("quantile", hi_q).fit(X, y).predict(Xt)), 0)
        test["cust_p50"] = np.where(open_, np.expm1(gbm().fit(X, yc).predict(Xt)), 0)

        fm = {"fold": i, **fold, "n_train": len(fit_rows), "seconds": round(time.time() - t0)}
        for col in ["baseline", "p50"]:
            fm[col] = metrics(test, col)
        o = test[open_ & (test.sales > 0)]
        fm["interval_coverage"] = float(((o.sales >= o.lo) & (o.sales <= o.hi)).mean())
        fold_metrics.append(fm)
        print(json.dumps(fm))
        preds.append(test[["store_id", "date", "week_start", "fold", "is_open", "promo",
                           "store_type", "sales", "customers",
                           "baseline", "p50", "lo", "hi", "cust_p50"]])

    fc = pd.concat(preds, ignore_index=True)
    fc.to_parquet(config.path("outputs") / "forecasts.parquet", index=False)

    overall = {col: metrics(fc, col) for col in ["baseline", "p50"]}
    o = fc[(fc.is_open == 1) & (fc.sales > 0)]
    overall["interval_coverage"] = float(((o.sales >= o.lo) & (o.sales <= o.hi)).mean())
    overall["nominal_coverage"] = hi_q - lo_q
    by_type = {t: metrics(g, "p50")["wape"] for t, g in fc.groupby("store_type")}
    # Weekly store-level WAPE — the grain sales managers actually look at.
    wk = fc.groupby(["store_id", "week_start"])[["sales", "p50", "baseline"]].sum()
    wk = wk[wk.sales > 0]
    weekly = {c: float(np.abs(wk[c] - wk.sales).sum() / wk.sales.sum()) for c in ["baseline", "p50"]}

    result = {"folds": fold_metrics, "overall_daily": overall, "wape_by_store_type": by_type,
              "weekly_store_wape": weekly}
    (config.path("reports") / "m2_backtest_metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in r if k != "folds"}, indent=2))
