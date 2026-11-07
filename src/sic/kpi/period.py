"""KPIs for any period (week, month, quarter), for the chain or one store.

Same rules as the weekly scorecard (src/sic/kpi/weekly.py):
  vs forecast   only days that have an out-of-sample forecast
  vs last year  day-matched: each day vs the same weekday 364 days earlier, same store, both
                open; the chain view excludes stores with a data gap so the store set is
                like-for-like
"""

from __future__ import annotations

import duckdb
import pandas as pd

from sic import config

SQL = """
with d as (
    select f.store_id, f.date, f.is_open, f.sales, s.has_data_gap
    from wh.marts.fct_daily_store_sales f join wh.marts.dim_store s using (store_id)
    where f.date between ? and ? {store}),
x as (
    select d.*, fc.p50, y.sales s364, y.is_open o364
    from d
    left join fc on fc.store_id = d.store_id and fc.date = d.date
    left join wh.marts.fct_daily_store_sales y
        on y.store_id = d.store_id and y.date = d.date - interval 364 day)
select
    sum(sales) filter (where is_open)                                        sales,
    sum(sales) filter (where is_open and p50 is not null)
      / sum(p50) filter (where is_open and p50 is not null) - 1              vs_forecast,
    sum(sales) filter (where is_open and o364 {gap})
      / sum(s364) filter (where is_open and o364 {gap}) - 1                  vs_last_year,
    count(distinct date)                                                     n_days
from x
"""


def kpis(start, end, store_id: int | None = None) -> dict:
    con = duckdb.connect()
    try:
        con.execute(f"attach '{config.path('warehouse')}' as wh (read_only)")
        con.execute("create view fc as select store_id, cast(date as date) date, p50 from "
                    f"read_parquet('{config.path('outputs') / 'forecasts_calibrated.parquet'}')")
        store = "and f.store_id = ?" if store_id is not None else ""
        gap = "" if store_id is not None else "and not has_data_gap"
        args = [pd.Timestamp(start).date(), pd.Timestamp(end).date()]
        args += [int(store_id)] if store_id is not None else []
        r = con.execute(SQL.format(store=store, gap=gap), args).df().iloc[0]
    finally:
        con.close()
    return {k: (None if pd.isna(v) else float(v)) for k, v in r.items()}


def periods(grain: str) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """(start, end, label) for each week / month / quarter that has forecasts, newest first.
    Partial periods are labelled with the days they actually cover."""
    f = pd.read_parquet(config.path("outputs") / "forecasts_calibrated.parquet",
                        columns=["date"]).date.drop_duplicates()
    freq = {"week": "W-SUN", "month": "M", "quarter": "Q"}[grain]
    out = []
    for p, days in f.groupby(f.dt.to_period(freq)):
        lo, hi = days.min(), days.max()
        full = lo == p.start_time.normalize() and hi == p.end_time.normalize()
        if grain == "week":
            end = hi if not full else p.end_time.normalize()
            label = (f"{lo:%-d}–{end:%-d %b %Y}" if lo.month == end.month
                     else f"{lo:%-d %b}–{end:%-d %b %Y}")
        elif grain == "month":
            label = f"{lo:%b %Y}" if full else f"{lo:%-d}–{hi:%-d %b %Y}"
        else:
            span = f"{lo:%b}" if lo.month == hi.month else f"{lo:%b}–{hi:%b}"
            label = f"Q{p.quarter} {p.year}" + ("" if full else f" ({span})")
        out.append((lo, hi, label))
    return out[::-1]
