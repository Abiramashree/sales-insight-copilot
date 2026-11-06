"""M3: root-cause analysis for detected anomalies.

Three lenses, each producing structured, auditable evidence (later the LLM
layer narrates this evidence; it never computes numbers itself):

1. Traffic vs basket   ln(S_act/S_exp) = ln(C_act/C_exp) + ln(B_act/B_exp)
2. Dimensional drill   Adtributor-style: for each dimension, how much of the
                       gap each segment explains, and how surprising its share is
3. Event evidence      store-level facts that plausibly explain an episode
                       (reopening, competitor opening, Promo2 start, holidays,
                       segment-wide movement, persistence after the episode)

Writes data/outputs/{episode_rca,segment_rca}.parquet and reports/m3_rca_examples.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sic import config

EASTER = [pd.Timestamp(d) for d in ("2013-03-31", "2014-04-20", "2015-04-05")]


# ---------- lens 1: traffic vs basket ----------
def traffic_basket(sales, forecast, customers, forecast_customers) -> dict:
    total = np.log(sales / forecast)
    traffic = np.log(customers / forecast_customers)
    basket = total - traffic
    share = traffic / total if abs(total) > 1e-9 else np.nan
    return {
        "total_log_gap": float(total),
        "traffic_log_gap": float(traffic),
        "basket_log_gap": float(basket),
        "traffic_share": float(np.clip(share, -1, 2)),
        "driver": "traffic" if share >= 0.6 else "basket" if share <= 0.4 else "mixed",
    }


# ---------- lens 2: dimensional drill-down ----------
def _js(p: float, q: float) -> float:
    """Jensen-Shannon surprise of one element (Adtributor)."""
    m = (p + q) / 2
    t = lambda a: 0.0 if a <= 0 else a * np.log(a / m)
    return 0.5 * (t(p) + t(q))


def drill_down(rows: pd.DataFrame, dims: list[str], max_elements: int = 3) -> list[dict]:
    """rows: one row per store (or store-week) with `sales` and `forecast`."""
    gap = rows.sales.sum() - rows.forecast.sum()
    results = []
    for d in dims:
        g = rows.groupby(d, observed=True)[["sales", "forecast"]].sum()
        g["gap"] = g.sales - g.forecast
        g["contribution"] = g.gap / gap
        g["surprise"] = [
            _js(f / rows.forecast.sum(), s / rows.sales.sum()) for f, s in zip(g.forecast, g.sales)
        ]
        g = g.sort_values("surprise", ascending=False)
        top = g[g.contribution > 0].head(max_elements)
        results.append({
            "dimension": d,
            "explanatory_power": float(top.contribution.sum()),
            "total_surprise": float(top.surprise.sum()),
            "elements": [
                {"value": str(k), "gap": float(r.gap), "gap_pct": float(r.sales / r.forecast - 1),
                 "contribution": float(r.contribution)}
                for k, r in top.iterrows()
            ],
        })
    return sorted(results, key=lambda r: r["total_surprise"], reverse=True)


# ---------- lens 3: event evidence for store episodes ----------
def episode_evidence(e: pd.Series, store: pd.Series, reopen: pd.Series,
                     w: pd.DataFrame) -> list[dict]:
    ev = []
    start, end = e.start, e.end
    lookback = start - pd.Timedelta(days=90)

    r = reopen.get(e.store_id)
    if r is not None and pd.notna(r) and lookback <= r <= end:
        ev.append({"type": "reopened_after_closure", "date": str(r.date()),
                   "detail": "Store reopened after >=7 days closed (e.g. refurbishment); "
                             "demand after reopening often differs from the pre-closure "
                             "level the model learned."})
    c = store.competition_open_date
    if pd.notna(c) and lookback <= pd.Timestamp(c) <= end:
        ev.append({"type": "competitor_opened", "date": str(pd.Timestamp(c).date()),
                   "detail": f"Competitor opened {store.competition_distance_m:.0f} m away."})
    p = store.promo2_start_date
    if pd.notna(p) and lookback <= pd.Timestamp(p) <= end:
        ev.append({"type": "promo2_started", "date": str(pd.Timestamp(p).date()),
                   "detail": "Store enrolled in the continuing Promo2 program."})
    if any(start - pd.Timedelta(days=7) <= d <= end for d in EASTER):
        ev.append({"type": "holiday_easter", "detail": "Episode overlaps Easter week."})
    if any(pd.Timestamp(y, 12, 1) <= end and start <= pd.Timestamp(y, 12, 31)
           for y in (2013, 2014)):
        ev.append({"type": "holiday_christmas", "detail": "Episode overlaps December trading."})

    # Segment-wide movement: share of same-type stores anomalous in the same direction.
    seg = w[(w.store_type == e.store_type) & w.week_start.between(start, end)]
    if len(seg):
        share = ((seg.is_anomaly) & (seg.direction == e.direction)).mean()
        if share >= 0.15:
            ev.append({"type": "segment_wide", "share": float(share),
                       "detail": f"{share:.0%} of type-{e.store_type} store-weeks moved the "
                                 "same way: likely a chain/segment effect, not store-specific."})

    # Persistence: does the gap continue in the 4 weeks after the episode?
    after = w[(w.store_id == e.store_id) & (w.week_start > end)
              & (w.week_start <= end + pd.Timedelta(weeks=4))]
    if len(after) >= 2:
        post = after.sales.sum() / after.forecast.sum() - 1
        if np.sign(post) == e.direction and abs(post) >= 0.05:
            ev.append({"type": "persistent_shift", "post_gap_pct": float(post),
                       "detail": f"Gap persists after the episode ({post:+.0%}): possible "
                                 "structural level shift rather than a one-off."})
    return ev


def classify(ev: list[dict]) -> str:
    types = [x["type"] for x in ev]
    for t in ("reopened_after_closure", "competitor_opened", "promo2_started", "segment_wide"):
        if t in types:
            return t
    if any(t.startswith("holiday") for t in types):
        return "holiday_timing"
    if "persistent_shift" in types:
        return "unexplained_level_shift"
    return "unexplained"


def _black_friday(year: int) -> pd.Timestamp:
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    return nov[nov.day_of_week == 4][3]  # 4th Friday (day after US Thanksgiving)


def calendar_events(year: int) -> list[tuple[str, pd.Timestamp, str]]:
    """Named holidays / retail events: (name, date, kind). Used only to *suggest* a cause."""
    from sic.forecast.features import _easter

    return [
        ("Easter", _easter(year), "holiday_timing"),
        ("Christmas", pd.Timestamp(year, 12, 25), "holiday_timing"),
        ("New Year / Epiphany (6 Jan, some states)", pd.Timestamp(year, 1, 6), "post_holiday"),
        ("Black Friday", _black_friday(year), "retail_event"),
    ]


def classify_segment(week_start, breadth: float, dd: list[dict], by_day: pd.DataFrame,
                     gap_pct: float | None = None) -> dict:
    """Name the most likely cause of a segment-week anomaly, with the evidence used."""
    week_end = week_start + pd.Timedelta(days=6)
    concentrated = dd and dd[0]["total_surprise"] > 0.01
    g = by_day.gap_pct.dropna()
    # direction follows the WEEK's gap (one closure day can swing the daily average)
    up = gap_pct > 0 if gap_pct is not None else (g.mean() > 0 if len(g) else True)
    peak_day = (g.idxmax() if up else g.idxmin()) if len(g) else None
    peak = f"{'peaks' if up else 'is weakest'} on {peak_day:%a %d %b} ({g[peak_day]:+.0%})" \
        if peak_day is not None else ""
    events = [e for y in {week_start.year, week_end.year} for e in calendar_events(y)
              if week_start - pd.Timedelta(days=4) <= e[1] <= week_end + pd.Timedelta(days=4)]
    if events and breadth >= 0.7:
        name, date, kind = events[0]
        why = {"holiday_timing": "Holiday-timing effects (stock-up before, dip around closures) "
                                 "are only partly captured by the forecaster's holiday features.",
               "post_holiday": "Typical post-holiday slump; regional holiday closures also "
                               "reduce trading days.",
               "retail_event": "The forecaster has no feature for this retail event; treat it as "
                               "a likely, not confirmed, driver."}[kind]
        return {"label": kind, "event": name,
                "detail": f"Broad-based ({breadth:.0%} of stores) and {peak}, around {name} "
                          f"({date:%d %b %Y}). {why}"}
    if breadth >= 0.7 and not concentrated:
        return {"label": "broad_based", "detail": f"{breadth:.0%} of stores moved together and "
                f"the week {peak}; no segment concentration — look for chain-wide drivers."}
    top = dd[0]["elements"][0] if dd else None
    return {"label": "concentrated", "detail": f"Concentrated in {dd[0]['dimension']} = "
            f"{top['value']} ({top['contribution']:.0%} of gap)." if top else "n/a"}


def run() -> dict:
    out = config.path("outputs")
    w = pd.read_parquet(out / "store_weeks.parquet")
    e = pd.read_parquet(out / "episodes.parquet")
    seg = pd.read_parquet(out / "segment_weeks.parquet")
    fc = pd.read_parquet(out / "forecasts_calibrated.parquet")
    con = config.connect(read_only=True)
    stores = con.execute("select * from marts.dim_store").df().set_index("store_id")
    # last reopening date per store (from feature logic: >=7-day gap between open days)
    od = con.execute("""select store_id, date from marts.fct_daily_store_sales
                        where is_open order by store_id, date""").df()
    con.close()
    od["date"] = pd.to_datetime(od.date)
    od["gap"] = od.groupby("store_id").date.diff().dt.days
    reopen_events = od[(od.gap >= 7) & (od.date >= "2014-05-01")]
    reopen = reopen_events.groupby("store_id").date.max()

    # --- store episodes ---
    rows = []
    for _, ep in e.iterrows():
        tb = traffic_basket(ep.sales, ep.forecast, ep.customers, ep.forecast_customers)
        ev = episode_evidence(ep, stores.loc[ep.store_id], reopen, w)
        rows.append({**ep.to_dict(), **{f"tb_{k}": v for k, v in tb.items()},
                     "root_cause": classify(ev), "evidence": json.dumps(ev)})
    erca = pd.DataFrame(rows)
    erca.to_parquet(out / "episode_rca.parquet", index=False)

    # --- segment anomalies: drill into stores, attributes, and days ---
    attrs = stores[["assortment", "competition_distance_filled_m", "promo2_enrolled",
                    "has_data_gap"]].copy()
    attrs["competition_band"] = pd.cut(attrs.competition_distance_filled_m,
                                       [0, 500, 2000, 8000, 1e9],
                                       labels=["<500m", "500m-2km", "2-8km", ">8km"])
    seg_rows = []
    for _, s in seg[seg.is_anomaly].iterrows():
        sw = w[(w.store_type == s.store_type) & (w.week_start == s.week_start)]
        sw = sw.join(attrs, on="store_id")
        days = fc[(fc.store_type == s.store_type) & (fc.week_start == s.week_start)]
        by_day = days.groupby("date")[["sales", "p50"]].sum()
        by_day["gap_pct"] = by_day.sales / by_day.p50.replace(0, np.nan) - 1
        tb = traffic_basket(sw.sales.sum(), sw.forecast.sum(), sw.customers.sum(),
                            sw.forecast_customers.sum())
        breadth = float((sw.gap_pct > 0).mean() if s.gap_pct > 0 else (sw.gap_pct < 0).mean())
        dd = drill_down(sw, ["assortment", "competition_band", "promo2_enrolled", "has_data_gap"])
        seg_rows.append({
            "root_cause": classify_segment(s.week_start, breadth, dd, by_day, s.gap_pct),
            "store_type": s.store_type, "week_start": s.week_start, "gap_pct": s.gap_pct,
            "robust_z": s.robust_z, "breadth_same_direction": breadth,
            **{f"tb_{k}": v for k, v in tb.items()},
            "drill_down": json.dumps(dd),
            "by_day": json.dumps({str(d.date()): None if np.isnan(g) else round(float(g), 3)
                                  for d, g in by_day.gap_pct.items()}),
            "top_stores": json.dumps(sw.nlargest(5, "gap")[["store_id", "gap", "gap_pct"]]
                                     .round(3).to_dict(orient="records")),
        })
    srca = pd.DataFrame(seg_rows)
    srca.to_parquet(out / "segment_rca.parquet", index=False)

    mix = erca.groupby(["root_cause", "direction"]).agg(
        episodes=("episode_id", "size"), total_gap=("gap", "sum")).reset_index()
    examples = {
        "root_cause_mix": mix.round(0).to_dict(orient="records"),
        "traffic_vs_basket_mix": erca.tb_driver.value_counts().to_dict(),
        "top_episodes": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.items()
             if k in ("store_id", "store_type", "direction", "start", "end", "weeks", "gap",
                      "gap_pct", "tb_driver", "tb_traffic_share", "root_cause")}
            | {"evidence": json.loads(r["evidence"])}
            for r in erca.head(8).to_dict(orient="records")
        ],
        "segment_anomalies": [
            {**{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.items()
                if k not in ("drill_down", "by_day", "top_stores")},
             "drill_down_top": json.loads(r["drill_down"])[:2],
             "by_day": json.loads(r["by_day"])}
            for r in srca.to_dict(orient="records")
        ],
    }
    (config.path("reports") / "m3_rca_examples.json").write_text(
        json.dumps(examples, indent=2, default=float))
    return examples


if __name__ == "__main__":
    ex = run()
    print(json.dumps({k: ex[k] for k in ("root_cause_mix", "traffic_vs_basket_mix")}, indent=2,
                     default=float))
    print(json.dumps(ex["top_episodes"][:4], indent=2, default=float))
    print(json.dumps(ex["segment_anomalies"], indent=2, default=float))
