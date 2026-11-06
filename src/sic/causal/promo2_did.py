"""M3: causal effect of enrolling a store in Promo2 — stacked difference-in-differences.

Design
  - Promo2 is opted into store-by-store with its own start week (staggered).
  - Each cohort (start month) forms a "stack": its newly enrolled stores plus all
    never-enrolled stores, observed in a window around the start. Stacking avoids
    the "forbidden comparisons" of naive two-way fixed effects with staggered timing.
  - Outcome: log(weekly sales per open day).
  - Fixed effects: stack x store, stack x week. SE clustered by store.
  - Event study in 4-week bins (reference = the 4 weeks before start) to check
    pre-trends; pooled post coefficient = the headline effect.
  - Placebo: pretend each cohort started `placebo_shift_weeks` earlier, using only
    pre-enrollment data. A credible design should find ~0 there.
  - Stores with a data gap (H2-2014 refurbishments) are excluded.

Writes reports/m3_promo2_did.json and data/outputs/promo2_event_study.parquet
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sic import config
from sic.causal.fe import fe_ols


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = config.connect(read_only=True)
    w = con.execute("""
        select w.store_id, w.week_start, w.sales, w.open_days
        from marts.fct_weekly_store_sales w join marts.dim_store s using (store_id)
        where not s.has_data_gap and w.open_days >= 4 and w.sales > 0""").df()
    s = con.execute("""
        select store_id, promo2_enrolled,
               cast(date_trunc('week', promo2_start_date) as date) start_week
        from marts.dim_store where not has_data_gap""").df()
    con.close()
    w["week_start"] = pd.to_datetime(w.week_start)
    s["start_week"] = pd.to_datetime(s.start_week)
    w["y"] = np.log(w.sales / w.open_days)
    return w, s


def build_stacks(w, s, pre: int, post: int, shift: int = 0, bin_w: int = 4,
                 min_pre: int = 26, min_post: int = 13,
                 treat_col: str = "promo2_enrolled") -> pd.DataFrame:
    """Generic stacked-DiD panel. `s` has store_id, start_week and a boolean `treat_col`
    (False = clean control). shift > 0 builds the placebo: fake start = real start -
    shift weeks, and only data before the real start is kept for treated stores."""
    first, last = w.week_start.min(), w.week_start.max()
    treated = s[s[treat_col] & s.start_week.notna()].copy()
    treated["fake_start"] = treated.start_week - pd.Timedelta(weeks=shift)
    treated = treated[(treated.fake_start - pd.Timedelta(weeks=min_pre) >= first)
                      & (treated.fake_start + pd.Timedelta(weeks=min_post) <= last)]
    if shift:
        treated = treated[treated.fake_start - pd.Timedelta(weeks=pre) >= first]
    treated["cohort"] = treated.fake_start.dt.to_period("M").astype(str)
    controls = s[~s[treat_col]].store_id

    stacks = []
    for cohort, tc in treated.groupby("cohort"):
        anchor = tc.fake_start.min()
        lo, hi = anchor - pd.Timedelta(weeks=pre), anchor + pd.Timedelta(weeks=post)
        if shift:  # placebo: never look past any real start in this cohort
            hi = min(hi, tc.start_week.min() - pd.Timedelta(days=1))
        win = w[w.week_start.between(lo, hi)]
        t = win[win.store_id.isin(tc.store_id)].merge(
            tc[["store_id", "fake_start"]], on="store_id")
        t["event_week"] = ((t.week_start - t.fake_start).dt.days // 7).astype(int)
        t["treated"] = 1
        c = win[win.store_id.isin(controls)].assign(event_week=np.nan, treated=0)
        stacks.append(pd.concat([t.drop(columns="fake_start"), c]).assign(stack=cohort))
    st = pd.concat(stacks, ignore_index=True)
    st = st[(st.treated == 0) | st.event_week.between(-pre, post)]
    st["post"] = ((st.treated == 1) & (st.event_week >= 0)).astype(float)
    st["ebin"] = np.floor(st.event_week / bin_w)
    st["stack_store"] = st["stack"] + "_" + st.store_id.astype(str)
    st["stack_week"] = st["stack"] + "_" + st.week_start.astype(str)
    return st


def event_study(st: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    bins = sorted(b for b in st.ebin.dropna().unique() if b != -1)
    names = []
    for b in bins:
        n = f"bin_{int(b)}"
        st[n] = ((st.treated == 1) & (st.ebin == b)).astype(float)
        names.append(n)
    r = fe_ols(st, "y", names, ["stack_store", "stack_week"], "store_id")
    ci = r.ci()
    es = pd.DataFrame({"bin": bins, "coef": r.params.values, "lo": ci.lo.values,
                       "hi": ci.hi.values})
    es = pd.concat([es, pd.DataFrame([{"bin": -1.0, "coef": 0.0, "lo": 0.0, "hi": 0.0}])])
    pre_names = [n for n, b in zip(names, bins) if b < -1]
    chi2, dof = r.wald(pre_names)
    return es.sort_values("bin").reset_index(drop=True), chi2, dof


def pooled(st: pd.DataFrame, extra: list[str] | None = None) -> dict:
    r = fe_ols(st, "y", ["post", *(extra or [])], ["stack_store", "stack_week"], "store_id")
    b, se = float(r.params["post"]), float(r.se["post"])
    return {"log_effect": b, "se": se, "effect_pct": float(np.exp(b) - 1),
            "ci_pct": [float(np.exp(b - 1.96 * se) - 1), float(np.exp(b + 1.96 * se) - 1)],
            "nobs": r.nobs, "clusters": r.nclusters}


def run() -> dict:
    from scipy.stats import chi2 as chi2_dist

    cfg = config.load()["causal"]
    pre, post = (abs(x) for x in cfg["event_window_weeks"])
    w, s = load_panel()
    st = build_stacks(w, s, pre, post, bin_w=cfg["event_bin_weeks"],
                      min_pre=cfg["promo2_min_pre_weeks"], min_post=cfg["promo2_min_post_weeks"])
    main = pooled(st)
    es, chi2, dof = event_study(st)
    es.to_parquet(config.path("outputs") / "promo2_event_study.parquet", index=False)

    shift = cfg["placebo_shift_weeks"]
    pst = build_stacks(w, s, pre=pre, post=shift - 1, shift=shift, bin_w=cfg["event_bin_weeks"],
                       min_pre=cfg["promo2_min_pre_weeks"], min_post=1)
    placebo = pooled(pst)

    # Robustness to the pre-enrollment dip (stores enrolled while sliding):
    #  (a) trend-adjusted: allow treated stores their own linear trend in event time
    #  (b) donut: drop the 12 weeks right before enrollment, where the dip sits
    st["treated_trend"] = np.where(st.treated == 1, st.event_week, 0.0)
    trend_adj = pooled(st, extra=["treated_trend"])
    donut = pooled(st[~((st.treated == 1) & st.event_week.between(-12, -1))].copy())

    result = {
        "design": "stacked DiD, cohorts by start month, never-enrolled controls",
        "treated_stores": int(st[st.treated == 1].store_id.nunique()),
        "control_stores": int(st[st.treated == 0].store_id.nunique()),
        "cohorts": int(st["stack"].nunique()),
        "effect": main,
        "pretrend_test": {"chi2": chi2, "dof": dof,
                          "p_value": float(1 - chi2_dist.cdf(chi2, dof))},
        "placebo": {**placebo, "shift_weeks": shift,
                    "treated_stores": int(pst[pst.treated == 1].store_id.nunique())},
        "robustness": {"trend_adjusted": trend_adj, "donut_drop_12w_pre": donut},
        "event_study": es.round(4).to_dict(orient="records"),
    }
    (config.path("reports") / "m3_promo2_did.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
