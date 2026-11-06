"""M3: causal lift of the daily Promo, per store, plus what drives heterogeneity.

Identification. Promo runs on the same dates for every store, so cross-store
comparison can't identify it. But the schedule alternates roughly week-on /
week-off independently of local demand, so WITHIN a store, promo days vs.
non-promo days — after controlling for day-of-week, month, year, holidays and
adjacent closures — is a credible comparison.

Per store:  log(sales) ~ promo + dow + month + year + school_holiday
                         + state_holiday + closed_yesterday + closed_tomorrow
            (Mon-Fri open days only; HC1 robust SE)
Then:       store lift ~ store attributes   (WLS, weights = 1/se^2)

Writes data/outputs/promo_lift_by_store.parquet, reports/m3_promo_lift.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sic import config
from sic.forecast.features import load_base


def store_regression(g: pd.DataFrame) -> dict:
    X = pd.get_dummies(
        g[["dow", "month", "year"]].astype(str), drop_first=True, dtype=float
    ).assign(
        const=1.0, promo=g.promo.values, school_holiday=g.school_holiday.values,
        state_holiday=(g.state_holiday != "none").astype(float).values,
        closed_yesterday=g.closed_yesterday.values, closed_tomorrow=g.closed_tomorrow.values,
    )
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    y = np.log(g.sales.values)
    Xv = X.values
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    b = XtX_inv @ Xv.T @ y
    u = y - Xv @ b
    n, k = Xv.shape
    V = XtX_inv @ (Xv.T * u**2) @ Xv @ XtX_inv * n / (n - k)  # HC1
    j = list(X.columns).index("promo")
    return {"beta": b[j], "se": float(np.sqrt(V[j, j])), "n": n}


def run() -> dict:
    df = load_base()
    d = df[(df.is_open == 1) & (df.sales > 0) & (df.dow <= 5)]
    rows = []
    for sid, g in d.groupby("store_id"):
        r = store_regression(g)
        rows.append({"store_id": sid, **r})
    lift = pd.DataFrame(rows)
    lift["lift"] = np.exp(lift.beta) - 1
    lift["lift_lo"] = np.exp(lift.beta - 1.96 * lift.se) - 1
    lift["lift_hi"] = np.exp(lift.beta + 1.96 * lift.se) - 1

    con = config.connect(read_only=True)
    stores = con.execute("""
        select store_id, store_type, assortment, promo2_enrolled,
               competition_distance_filled_m, avg_daily_sales_open
        from marts.dim_store""").df()
    nonpromo = con.execute("""
        select store_id, avg(sales) avg_nonpromo_weekday_sales
        from marts.fct_daily_store_sales
        where is_open and not promo and dow <= 5 and date >= date '2015-01-01'
        group by 1""").df()
    con.close()
    lift = lift.merge(stores, on="store_id").merge(nonpromo, on="store_id", how="left")
    # € incremental per promo week (5 promo days) at the store's recent non-promo run-rate
    lift["incremental_eur_per_promo_week"] = lift.avg_nonpromo_weekday_sales * lift.lift * 5
    lift.to_parquet(config.path("outputs") / "promo_lift_by_store.parquet", index=False)

    # Heterogeneity: which store traits predict bigger lift?
    h = lift.assign(log_comp=np.log(lift.competition_distance_filled_m),
                    log_sales=np.log(lift.avg_daily_sales_open), w=1 / lift.se**2)
    het = smf.wls("beta ~ C(store_type) + C(assortment) + log_comp + promo2_enrolled + log_sales",
                  data=h, weights=h.w).fit(cov_type="HC1")
    het_tbl = pd.DataFrame({"coef": het.params, "se": het.bse, "p": het.pvalues}).round(4)

    pooled_w = 1 / lift.se**2
    pooled_beta = float((lift.beta * pooled_w).sum() / pooled_w.sum())
    result = {
        "n_stores": len(lift),
        "pooled_lift": float(np.exp(pooled_beta) - 1),
        "median_store_lift": float(lift.lift.median()),
        "lift_p10_p90": [float(lift.lift.quantile(0.1)), float(lift.lift.quantile(0.9))],
        "share_stores_ci_above_zero": float((lift.lift_lo > 0).mean()),
        "naive_raw_uplift_for_comparison": None,  # filled from EDA in the report
        "by_store_type": lift.groupby("store_type").lift.median().round(4).to_dict(),
        "by_assortment": lift.groupby("assortment").lift.median().round(4).to_dict(),
        "heterogeneity_model": {"r2": float(het.rsquared),
                                "coefficients": het_tbl.to_dict(orient="index")},
        "top_responders": lift.nlargest(5, "incremental_eur_per_promo_week")[
            ["store_id", "store_type", "lift", "lift_lo", "lift_hi",
             "incremental_eur_per_promo_week"]].round(3).to_dict(orient="records"),
        "weakest_responders": lift.nsmallest(5, "lift")[
            ["store_id", "store_type", "lift", "lift_lo", "lift_hi"]].round(3)
            .to_dict(orient="records"),
    }
    (config.path("reports") / "m3_promo_lift.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
