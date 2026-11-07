"""A/B experiments on stores: assign, read out, dry-run on history (no LLM).

assign()    reproducible stratified randomization: stores are sorted by pre-period weekly
            sales, paired with their neighbour, and one of each pair goes to test (seeded),
            so the groups start balanced.
analyze()   readout from weekly results (store_id, week_start, group, sales): the outcome
            is each store's mean log weekly sales in the test weeks; CUPED adjusts it by the
            store's own pre-period mean (same number of weeks before the test, from the
            warehouse). Welch test on store-level values -> lift, 95% CI, p-value, decision.
dry_run()   the same readout on real past weeks with no treatment applied (an A/A test): a
            sound design should show a lift near 0 with a CI covering 0.
read_csv()  validate an uploaded results file.

Decision rule (fixed before the readout): roll out if the CUPED 95% CI is above 0 and the
lift is at least half the lift the test was sized for; else don't roll out.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from sic import config

REQUIRED = ["store_id", "week_start", "group", "sales"]


def weekly() -> pd.DataFrame:
    """Store-week sales for complete weeks (>= 90% of the busiest week's store-days)."""
    con = config.connect(read_only=True)
    try:
        w = con.execute("""
            select store_id, week_start, sum(sales) sales, count(*) n_days
            from marts.fct_daily_store_sales where is_open and sales > 0
            group by 1, 2""").df()
    finally:
        con.close()
    w["week_start"] = pd.to_datetime(w.week_start)
    full = w.groupby("week_start").n_days.sum()
    return w[w.week_start.isin(full[full >= 0.9 * full.max()].index)]


def assign(store_ids: list[int], seed: int = 7, pre_weeks: int = 8) -> pd.DataFrame:
    """Test/control groups, stratified by recent weekly sales. Same input, same groups."""
    ids = sorted({int(i) for i in store_ids})
    if len(ids) < 4:
        raise ValueError("Need at least 4 stores to split into test and control.")
    w = weekly()
    recent = sorted(w.week_start.unique())[-pre_weeks:]
    base = w[w.week_start.isin(recent) & w.store_id.isin(ids)].groupby("store_id").sales.mean()
    base = base.reindex(ids).fillna(base.median()).sort_values(kind="stable")
    rng = np.random.default_rng(seed)
    group = {}
    order = list(base.index)
    for i in range(0, len(order) - 1, 2):
        a, b = order[i], order[i + 1]
        t = rng.random() < 0.5
        group[a], group[b] = ("test", "control") if t else ("control", "test")
    if len(order) % 2:
        group[order[-1]] = "test" if rng.random() < 0.5 else "control"
    return pd.DataFrame({"store_id": order, "group": [group[s] for s in order],
                         "weekly_sales": base.round(0).values})


def analyze(results: pd.DataFrame, planned_lift: float | None = None) -> dict:
    """Read out a store-level A/B test from weekly results."""
    r = results.copy()
    r["week_start"] = pd.to_datetime(r.week_start)
    r = r[r.sales > 0]
    start, weeks = r.week_start.min(), r.week_start.nunique()
    y = r.groupby(["store_id", "group"]).sales.apply(lambda s: np.log(s).mean()).reset_index()
    w = weekly()
    pre_w = sorted(w[w.week_start < start].week_start.unique())[-weeks:]
    x = w[w.week_start.isin(pre_w)].groupby("store_id").sales.apply(lambda s: np.log(s).mean())
    y = y.join(x.rename("pre"), on="store_id")
    have_pre = y.pre.notna().all() and len(pre_w) == weeks
    if have_pre:
        theta = np.cov(y.sales, y.pre, ddof=1)[0, 1] / y.pre.var(ddof=1)
        y["adj"] = y.sales - theta * (y.pre - y.pre.mean())
    out = {"test_start": str(start.date()), "weeks": int(weeks),
           "stores_test": int((y.group == "test").sum()),
           "stores_control": int((y.group == "control").sum()), "planned_lift": planned_lift,
           "cuped": bool(have_pre)}
    for k, col in (("raw", "sales"), ("cuped", "adj")):
        if col not in y:
            continue
        t, c = y[y.group == "test"][col], y[y.group == "control"][col]
        if len(t) < 2 or len(c) < 2:
            raise ValueError("Each group needs at least 2 stores with results.")
        d = t.mean() - c.mean()
        vt, vc = t.var(ddof=1) / len(t), c.var(ddof=1) / len(c)
        se = np.sqrt(vt + vc)
        df = (vt + vc) ** 2 / (vt ** 2 / (len(t) - 1) + vc ** 2 / (len(c) - 1))
        h = stats.t.ppf(0.975, df) * se
        out[k] = {"lift": float(np.expm1(d)), "ci": [float(np.expm1(d - h)), float(np.expm1(d + h))],
                  "p_value": float(stats.ttest_ind(t, c, equal_var=False).pvalue)}
    best = out.get("cuped") if have_pre else out["raw"]
    need = (planned_lift or 0) / 2
    out["decision"] = ("roll out" if best["ci"][0] > 0 and best["lift"] >= need
                       else "do not roll out")
    out["rule"] = ("Roll out if the 95% CI is above 0 and the lift is at least half the "
                   "planned lift" + (f" ({need:.1%})" if planned_lift else "") + ".")
    return out


def dry_run(groups: pd.DataFrame, start_week: str, weeks: int,
            planned_lift: float | None = None) -> dict:
    """Read out the design on real past weeks with no treatment (an A/A test)."""
    w = weekly()
    span = sorted(w[w.week_start >= pd.Timestamp(start_week)].week_start.unique())[:weeks]
    if len(span) < weeks:
        raise ValueError(f"Only {len(span)} complete weeks of data from {start_week}.")
    res = w[w.week_start.isin(span)].merge(groups[["store_id", "group"]], on="store_id")
    out = analyze(res[REQUIRED], planned_lift)
    out["dry_run"] = True
    return out


def read_csv(data: bytes | str) -> pd.DataFrame:
    """Validate an uploaded results file: store_id, week_start, group (test/control), sales."""
    df = pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data))
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)} (need {', '.join(REQUIRED)}).")
    df["group"] = df.group.astype(str).str.strip().str.lower()
    bad = set(df.group) - {"test", "control"}
    if bad:
        raise ValueError(f"group must be 'test' or 'control' (found {sorted(bad)[:3]}).")
    df["week_start"] = pd.to_datetime(df.week_start, errors="raise")
    df["sales"] = pd.to_numeric(df.sales, errors="raise")
    if df.groupby("store_id").group.nunique().max() > 1:
        raise ValueError("A store appears in both groups.")
    return df[REQUIRED]
