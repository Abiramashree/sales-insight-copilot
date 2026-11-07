"""Hypothesis tester and A/B test planner (no LLM).

test(): is a metric different between two conditions? The unit of analysis is the store,
never the store-day (thousands of correlated days would make anything look significant).

  within-store   promo vs no promo, school holiday vs not. Each store is compared with
                 itself on the same weekdays (mean log metric per store x weekday, A - B,
                 averaged over weekdays), then a paired t-test across stores. Store traits
                 cancel out; timing can still confound (e.g. promos placed in strong weeks).
  between-store  store type, assortment, Promo2, competitor distance. Welch t-test on each
                 store's mean log metric. Stores differ in other ways too: an association,
                 not a causal effect.

Effects are on the log scale and reported as % (exp(d) - 1); a rank test (Wilcoxon /
Mann-Whitney) is reported alongside as a check that one outlier store isn't driving it.

plan(): minimum detectable effect and stores needed for a store-level A/B test, using the
real between-store spread of weekly log sales, with and without CUPED (adjusting each
store's test-period outcome by its own pre-period, Deng et al. 2013).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from sic import config

METRICS = {"sales": "Sales per open day", "customers": "Customers per open day",
           "basket": "Basket (sales per customer)"}
WITHIN = {"promo": ("Promo", "No promo"), "school_holiday": ("School holiday", "No school holiday")}
BETWEEN = {"store_type": "Store type", "assortment": "Assortment", "promo2": "Promo2",
           "competitor": "Nearest competitor"}


ATTRS = """s.store_type, s.assortment,
    case when s.promo2_enrolled then 'Enrolled' else 'Not enrolled' end promo2,
    case when s.competition_distance_m < 1000 then 'Within 1 km'
         when s.competition_distance_m >= 1000 then '1 km or more' end competitor"""


def daily(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Open store-days with sales, customers and the store attributes tests can split on."""
    where = " and ".join(["f.is_open", "f.sales > 0", "f.customers > 0"]
                         + ([f"f.date >= date '{start}'"] if start else [])
                         + ([f"f.date <= date '{end}'"] if end else []))
    con = config.connect(read_only=True)
    try:
        d = con.execute(f"""
            select f.store_id, f.date, f.dow, f.sales, f.customers, f.promo::int promo,
                   f.school_holiday::int school_holiday, {ATTRS}
            from marts.fct_daily_store_sales f join marts.dim_store s using (store_id)
            where {where}""").df()
    finally:
        con.close()
    d["y_sales"], d["y_customers"] = np.log(d.sales), np.log(d.customers)
    d["y_basket"] = d.y_sales - d.y_customers
    return d


def _t_ci(diff: float, se: float, df: float) -> tuple[float, float]:
    h = stats.t.ppf(0.975, df) * se
    return diff - h, diff + h


def within(d: pd.DataFrame, metric: str, flag: str) -> dict:
    """Paired, weekday-matched comparison of flag == 1 vs 0 within each store."""
    y = f"y_{metric}"
    m = d.groupby(["store_id", "dow", flag])[y].mean().unstack(flag).dropna()
    per_store = (m[1] - m[0]).groupby("store_id").mean()
    n = len(per_store)
    if n < 3:
        raise ValueError("Fewer than 3 stores have both conditions on the same weekdays.")
    diff, se = per_store.mean(), per_store.std(ddof=1) / np.sqrt(n)
    lo, hi = _t_ci(diff, se, n - 1)
    return {"design": "within-store", "a": WITHIN[flag][0], "b": WITHIN[flag][1],
            "effect_pct": float(np.expm1(diff)), "ci_pct": [float(np.expm1(lo)),
                                                            float(np.expm1(hi))],
            "p_value": float(stats.ttest_1samp(per_store, 0).pvalue),
            "p_rank": float(stats.wilcoxon(per_store).pvalue),
            "n_a": n, "n_b": n, "per_store": np.expm1(per_store).rename("effect").reset_index()}


def between(d: pd.DataFrame, metric: str, by: str, a: str, b: str) -> dict:
    """Welch comparison of store-level means between group a and group b."""
    y = f"y_{metric}"
    s = d.groupby(["store_id", by])[y].mean().reset_index()
    ya, yb = s[s[by] == a][y], s[s[by] == b][y]
    if len(ya) < 3 or len(yb) < 3:
        raise ValueError(f"Need at least 3 stores in each group ({len(ya)} vs {len(yb)}).")
    diff = ya.mean() - yb.mean()
    va, vb = ya.var(ddof=1) / len(ya), yb.var(ddof=1) / len(yb)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(ya) - 1) + vb ** 2 / (len(yb) - 1))
    lo, hi = _t_ci(diff, se, df)
    return {"design": "between-store", "a": a, "b": b,
            "effect_pct": float(np.expm1(diff)), "ci_pct": [float(np.expm1(lo)),
                                                            float(np.expm1(hi))],
            "p_value": float(stats.ttest_ind(ya, yb, equal_var=False).pvalue),
            "p_rank": float(stats.mannwhitneyu(ya, yb).pvalue),
            "n_a": len(ya), "n_b": len(yb),
            "per_store": s.rename(columns={by: "group", y: "value"}).assign(
                value=lambda x: np.exp(x.value))}


def test(metric: str, comparison: str, a: str | None = None, b: str | None = None,
         store_type: str | None = None, start: str | None = None,
         end: str | None = None) -> dict:
    if metric not in METRICS or comparison not in {**WITHIN, **BETWEEN}:
        raise ValueError(f"metric in {list(METRICS)}, comparison in "
                         f"{list(WITHIN) + list(BETWEEN)}")
    d = daily(start, end)
    if store_type:
        d = d[d.store_type == store_type]
    r = within(d, metric, comparison) if comparison in WITHIN else between(
        d, metric, comparison, a, b)
    r.update({"metric": metric, "comparison": comparison, "store_type": store_type,
              "period": [str(d.date.min().date()), str(d.date.max().date())]})
    return r


def groups(comparison: str) -> list[str]:
    """The values a between-store comparison can take (same definitions as daily())."""
    if comparison not in BETWEEN:
        raise ValueError(f"Unknown comparison {comparison!r}; choose from {list(BETWEEN)}.")
    con = config.connect(read_only=True)
    try:
        g = con.execute(f"select distinct {comparison} from (select {ATTRS} "
                        f"from marts.dim_store s) where {comparison} is not null").fetchall()
    finally:
        con.close()
    return sorted(v for (v,) in g)


# ------------------------------------------------------------------ planner
def weekly_spread(weeks: int, metric: str = "sales") -> dict:
    """Between-store SD of the mean weekly log metric over the last `weeks` complete weeks,
    and of the CUPED-adjusted version (pre-period = the `weeks` weeks before)."""
    con = config.connect(read_only=True)
    try:
        w = con.execute("""
            select store_id, week_start, sum(sales) sales, sum(customers) customers,
                   count(*) n_days
            from marts.fct_daily_store_sales where is_open and sales > 0
            group by 1, 2""").df()
    finally:
        con.close()
    full = w.groupby("week_start").n_days.sum()
    w = w[w.week_start.isin(full[full >= 0.9 * full.max()].index)]    # complete weeks only
    col = {"sales": "sales", "customers": "customers"}.get(metric, "sales")
    w["y"] = np.log(w[col] / w.n_days)
    last = sorted(w.week_start.unique())
    test_w, pre_w = last[-weeks:], last[-2 * weeks:-weeks]
    y = w[w.week_start.isin(test_w)].groupby("store_id").y.mean()
    x = w[w.week_start.isin(pre_w)].groupby("store_id").y.mean()
    yx = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    theta = np.cov(yx.y, yx.x, ddof=1)[0, 1] / yx.x.var(ddof=1)
    adj = yx.y - theta * (yx.x - yx.x.mean())
    return {"sd_raw": float(yx.y.std(ddof=1)), "sd_cuped": float(adj.std(ddof=1)),
            "stores": len(yx), "weeks": weeks}


def mde(sd: float, stores_per_arm, alpha: float = 0.05, power: float = 0.8):
    """Minimum detectable lift (as a fraction) for a two-group store-level test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return np.expm1(z * sd * np.sqrt(2 / np.asarray(stores_per_arm, dtype=float)))


def plan(weeks: int, stores_per_arm: int, target_lift: float, alpha: float = 0.05,
         power: float = 0.8, metric: str = "sales", spread: dict | None = None) -> dict:
    """MDE for the given design and stores per arm needed to detect `target_lift`.
    `spread` (from weekly_spread) can be passed in to avoid recomputing it."""
    sp = spread or weekly_spread(weeks, metric)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    out = {"weeks": weeks, "stores_per_arm": stores_per_arm, "target_lift": target_lift,
           "alpha": alpha, "power": power, "available_stores": sp["stores"]}
    for k in ("raw", "cuped"):
        sd = sp[f"sd_{k}"]
        out[f"mde_{k}"] = float(mde(sd, stores_per_arm, alpha, power))
        out[f"stores_needed_{k}"] = int(np.ceil(2 * (z * sd / np.log1p(target_lift)) ** 2))
    return out
