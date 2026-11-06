"""Weekly KPI scorecard: chain + store-type level, every week of the forecast period.

Comparisons are day-matched (each day vs the same weekday 7 / 364 days earlier, same
store, both open), so partial weeks and refurbishment closures don't distort WoW/YoY.
Definitions: sql/kpis.yaml. Writes data/outputs/weekly_kpis.parquet and
reports/weekly_kpis.md (+ figures).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from sic import config
from sic.plotting import BLUE, NEUTRAL, ORANGE, TEXT_2, eur, save

DAILY_SQL = """
with d as (
    select f.store_id, f.date, f.week_start, f.dow, f.is_open, f.sales, f.customers, f.promo,
           s.store_type, s.has_data_gap
    from wh.marts.fct_daily_store_sales f join wh.marts.dim_store s using (store_id)),
x as (
    select d.*, fc.p50,
           p.sales s7, p.is_open o7,
           y.sales s364, y.customers c364, y.is_open o364
    from d
    left join fc on fc.store_id = d.store_id and fc.date = d.date
    left join d p on p.store_id = d.store_id and p.date = d.date - interval 7 day
    left join d y on y.store_id = d.store_id and y.date = d.date - interval 364 day
    where d.week_start in (select distinct fc.week_start from fc))
select week_start, coalesce(store_type, 'ALL') segment,
    sum(sales) filter (where is_open)                                  sales,
    sum(customers) filter (where is_open)                              customers,
    sum(sales) filter (where is_open) / sum(customers) filter (where is_open) basket,
    sum(sales) filter (where is_open) / count(*) filter (where is_open) sales_per_store_day,
    count(distinct store_id) filter (where is_open)                    stores_reporting,
    count(*) filter (where is_open)                                    open_store_days,
    sum(sales) filter (where promo and is_open) / sum(sales) filter (where is_open)
                                                                       promo_sales_share,
    bool_or(promo)                                                     promo_week,
    avg(case when is_open then 0 else 1 end) filter (where dow < 7)    closure_rate,
    sum(sales) filter (where is_open) / sum(p50) filter (where is_open) - 1 sales_vs_forecast,
    sum(p50) filter (where is_open)                                    forecast,
    sum(sales) filter (where is_open and o7) / sum(s7) filter (where is_open and o7) - 1
                                                                       wow_growth,
    sum(sales) filter (where is_open and o364 and not has_data_gap)
      / sum(s364) filter (where is_open and o364 and not has_data_gap) - 1  yoy_growth,
    sum(customers) filter (where is_open and o364 and not has_data_gap)
      / sum(c364) filter (where is_open and o364 and not has_data_gap) - 1  yoy_customers_growth
from x
group by grouping sets ((week_start), (week_start, store_type))
order by week_start, segment
"""


def compute() -> pd.DataFrame:
    import duckdb

    out = config.path("outputs")
    con = duckdb.connect()
    con.execute(f"attach '{config.path('warehouse')}' as wh (read_only)")
    con.execute(f"create view fc as select store_id, cast(date as date) date, "
                f"cast(week_start as date) week_start, p50 "
                f"from read_parquet('{out / 'forecasts_calibrated.parquet'}')")
    k = con.execute(DAILY_SQL).df()
    k["week_start"] = pd.to_datetime(k.week_start)
    k["yoy_basket_growth"] = (1 + k.yoy_growth) / (1 + k.yoy_customers_growth) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        k["yoy_traffic_share"] = np.log1p(k.yoy_customers_growth) / np.log1p(k.yoy_growth)

    # store-week grain KPIs (anomalies, model health, concentration)
    w = pd.read_parquet(out / "store_weeks.parquet")

    def sw(g: pd.DataFrame) -> pd.Series:
        ab = (g.sales - g.forecast).abs()
        return pd.Series({
            "anomaly_rate": g.is_anomaly.mean(),
            "stores_below_forecast": int((g.is_anomaly & (g.direction < 0)).sum()),
            "stores_above_forecast": int((g.is_anomaly & (g.direction > 0)).sum()),
            "forecast_wape": ab.sum() / g.sales.sum(),
            "forecast_bias": (g.forecast.sum() - g.sales.sum()) / g.sales.sum(),
            "gap_concentration_top10": ab.nlargest(10).sum() / ab.sum(),
        })

    chain = w.groupby("week_start").apply(sw, include_groups=False).assign(segment="ALL")
    seg = w.groupby(["week_start", "store_type"]).apply(sw, include_groups=False) \
        .reset_index("store_type").rename(columns={"store_type": "segment"})
    sk = pd.concat([chain, seg]).reset_index()
    k = k.merge(sk, on=["week_start", "segment"], how="left")

    k = k.sort_values(["segment", "week_start"])
    roll = k.groupby("segment")[["sales", "forecast"]].transform(
        lambda s: s.rolling(4, min_periods=4).sum())
    k["sales_vs_forecast_4wk"] = roll.sales / roll.forecast - 1
    k["partial_week"] = k.open_store_days < k.groupby("segment").open_store_days.transform(
        "median") * 0.85
    return k.sort_values(["week_start", "segment"]).reset_index(drop=True)


SCORECARD = [  # (column, label, kind)
    ("sales", "Sales", "eur"),
    ("sales_vs_forecast", "Sales vs forecast", "pct"),
    ("sales_vs_forecast_4wk", "Sales vs forecast, 4-wk", "pct"),
    ("wow_growth", "Week-over-week (day-matched)", "pct"),
    ("yoy_growth", "YoY same-store (day-matched)", "pct"),
    ("yoy_customers_growth", "  …customers YoY", "pct"),
    ("yoy_basket_growth", "  …basket YoY", "pct"),
    ("customers", "Customer visits", "int"),
    ("basket", "Avg basket", "eur2"),
    ("sales_per_store_day", "Sales per open store-day", "eur"),
    ("promo_sales_share", "Promo sales share", "pct0"),
    ("closure_rate", "Mon–Sat closure rate", "pct1"),
    ("stores_reporting", "Stores reporting", "int"),
    ("anomaly_rate", "Anomaly rate", "pct1"),
    ("stores_below_forecast", "  …stores below forecast", "int"),
    ("stores_above_forecast", "  …stores above forecast", "int"),
    ("gap_concentration_top10", "Gap from top-10 stores", "pct0"),
    ("forecast_wape", "Forecast error (WAPE)", "pct1"),
    ("forecast_bias", "Forecast bias (+ = over-forecast)", "pct1"),
]


def fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return {"eur": lambda: eur(v), "eur2": lambda: f"€{v:.2f}", "int": lambda: f"{v:,.0f}",
            "pct": lambda: f"{v * 100:+.1f}%", "pct0": lambda: f"{v * 100:.0f}%",
            "pct1": lambda: f"{v * 100:.1f}%"}[kind]()


def figures(k: pd.DataFrame) -> dict:
    c = k[k.segment == "ALL"]
    f = {}
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.plot(c.week_start, c.sales_vs_forecast, color=NEUTRAL, linewidth=1.5, label="Weekly")
    ax.plot(c.week_start, c.sales_vs_forecast_4wk, color=BLUE, label="Rolling 4 weeks")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Chain sales vs forecast")
    ax.legend(loc="upper left")
    f["vs_forecast"] = save(fig, "kpi_sales_vs_forecast")

    cy = c.dropna(subset=["yoy_growth"])
    cy = cy[~cy.partial_week]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(cy))
    ax.bar(x - 0.21, cy.yoy_customers_growth, 0.4, color=BLUE, label="Customers YoY")
    ax.bar(x + 0.21, cy.yoy_basket_growth, 0.4, color=ORANGE, label="Basket YoY")
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    step = max(1, len(cy) // 10)
    ax.set_xticks(x[::step], [d.strftime("%d %b %y") for d in cy.week_start][::step])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Same-store YoY growth, split into customer visits and basket size")
    ax.legend(loc="upper left")
    f["yoy"] = save(fig, "kpi_yoy_decomposition")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(c.week_start, c.forecast_wape, color=BLUE)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(bottom=0)
    ax.set_title("Model health: weekly forecast error (store-week WAPE)")
    f["wape"] = save(fig, "kpi_forecast_wape")
    return f


def report(k: pd.DataFrame) -> str:
    c = k[k.segment == "ALL"].reset_index(drop=True)
    last, prev = c.iloc[-1], c.iloc[-2]
    full = c[~c.partial_week]
    avg4 = full.tail(4)
    f = figures(k)
    rows = "\n".join(
        f"| {lab} | **{fmt(last[col], kind)}** | {fmt(prev[col], kind)} | "
        f"{fmt(avg4[col].mean(), kind)} |"
        for col, lab, kind in SCORECARD)
    segs = k[(k.week_start == last.week_start) & (k.segment != "ALL")]
    seg_rows = "\n".join(
        f"| {r.segment} | {fmt(r.sales, 'eur')} | {fmt(r.sales_vs_forecast, 'pct')} | "
        f"{fmt(r.yoy_growth, 'pct')} | {fmt(r.basket, 'eur2')} | {fmt(r.promo_sales_share, 'pct0')} | "
        f"{fmt(r.anomaly_rate, 'pct1')} |" for r in segs.itertuples())
    partial = " (partial week: data ends Fri 2015-07-31; comparisons are day-matched)" \
        if last.partial_week else ""
    md = f"""# Weekly KPI report — week of {last.week_start:%Y-%m-%d}{partial}

Definitions: [sql/kpis.yaml](../sql/kpis.yaml). WoW and YoY compare each day with the same
weekday 7 / 364 days earlier (same store, both open); YoY excludes the 180 stores offline
in H2-2014. Promo week: **{'yes' if last.promo_week else 'no'}** (prior week:
{'yes' if prev.promo_week else 'no'}). Promo weeks alternate, so WoW swings are expected.

## Scorecard

| KPI | This week | Prior week | 4-week avg (full weeks) |
|---|---|---|---|
{rows}

## By store type (this week)

| Type | Sales | vs forecast | YoY same-store | Basket | Promo share | Anomaly rate |
|---|---|---|---|---|---|---|
{seg_rows}

## Trends

![vs forecast](../{f['vs_forecast']})

![yoy](../{f['yoy']})

![wape](../{f['wape']})
"""
    p = config.path("reports") / "weekly_kpis.md"
    p.write_text(md)
    return str(p)


def run() -> dict:
    k = compute()
    k.to_parquet(config.path("outputs") / "weekly_kpis.parquet", index=False)
    report(k)
    last = k[k.segment == "ALL"].iloc[-1]
    return {c: (None if pd.isna(last[c]) else (float(last[c]) if not isinstance(last[c], str)
                                              else last[c]))
            for c, _, _ in SCORECARD} | {"week_start": str(last.week_start.date())}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
