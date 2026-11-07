"""Fact packs: the ONLY numbers an LLM is allowed to use.

Each fact = {id, label, value, display}. `display` is pre-formatted so the
model copies strings instead of doing arithmetic; `value` is what the
fact-checker compares against.
"""

from __future__ import annotations

import json

import pandas as pd

from sic import config
from sic.insights.factcheck import numbers_in


class FactPack:
    def __init__(self, subject: str, kind: str):
        self.subject, self.kind = subject, kind
        self.facts: list[dict] = []
        self.context: dict = {}

    def add(self, label: str, value, display: str | None = None) -> str:
        fid = f"f{len(self.facts) + 1}"
        if display is None:
            display = str(value)
        self.facts.append({"id": fid, "label": label, "value": value, "display": display})
        return fid

    def values(self) -> list[float]:
        out = []
        for f in self.facts:
            v = f["value"]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
            elif isinstance(v, list):
                out.extend(float(x) for x in v if isinstance(x, (int, float)))
            # numbers quoted inside display text or labels ("95% CI") are citable too
            out.extend(numbers_in(f["display"]))
            out.extend(numbers_in(f["label"]))
        return out

    def to_prompt(self) -> str:
        return json.dumps({"subject": self.subject, "kind": self.kind,
                           "facts": [{k: f[k] for k in ("id", "label", "display")}
                                     for f in self.facts],
                           "context": self.context}, indent=1, default=str)


def pct(x, d=0) -> str:
    return f"{x * 100:+.{d}f}%"


def eur(x) -> str:
    a = abs(x)
    s = f"€{a / 1e6:.2f}M" if a >= 1e6 else f"€{a / 1e3:.1f}k" if a >= 1e4 else f"€{a:,.0f}"
    return ("-" if x < 0 else "") + s


class Store:
    """Lazy access to pipeline outputs."""

    def __init__(self):
        out, rep = config.path("outputs"), config.path("reports")
        self.erca = pd.read_parquet(out / "episode_rca.parquet")
        self.recs = pd.read_parquet(out / "recommendations.parquet")
        self.lift = pd.read_parquet(out / "promo_lift_by_store.parquet").set_index("store_id")
        self.weeks = pd.read_parquet(out / "store_weeks.parquet")
        self.segments = pd.read_parquet(out / "segment_rca.parquet")
        self.did = json.loads((rep / "m3_promo2_did.json").read_text())
        self.pl = json.loads((rep / "m3_promo_lift.json").read_text())
        self.bt = json.loads((rep / "m2_backtest_metrics.json").read_text())
        kp = out / "weekly_kpis.parquet"
        self.kpis = pd.read_parquet(kp) if kp.exists() else pd.DataFrame(
            columns=["segment", "week_start"])


def chain_week(s: Store) -> FactPack:
    w = s.weeks
    last = w.week_start.max()
    cur = w[w.week_start == last]
    prev = w[w.week_start == last - pd.Timedelta(weeks=1)]
    fp = FactPack(f"Chain performance, week of {last:%Y-%m-%d}", "chain_weekly")
    fp.add("Chain sales this week", float(cur.sales.sum()), eur(cur.sales.sum()))
    g = cur.sales.sum() / cur.forecast.sum() - 1
    fp.add("Chain sales vs forecast", float(g), pct(g, 1))
    # day-matched (same stores, same weekdays): the last week is partial (Mon-Fri), so raw
    # week totals are not comparable. Falls back to raw totals if KPIs are missing.
    kw = s.kpis[(s.kpis.segment == "ALL") & (s.kpis.week_start == last)]
    if len(kw) and pd.notna(kw.iloc[0].get("wow_growth")):
        wow = float(kw.iloc[0].wow_growth)
    else:
        wow = cur.sales.sum() / prev.sales.sum() - 1 if len(prev) else 0.0
    fp.add("Chain sales vs prior week", float(wow), pct(wow, 1))
    fp.add("Stores flagged anomalous this week", int(cur.is_anomaly.sum()))
    fp.add("…of which below forecast", int((cur.is_anomaly & (cur.direction < 0)).sum()))
    fp.add("…of which above forecast", int((cur.is_anomaly & (cur.direction > 0)).sum()))
    fp.add("Stores reporting this week", int(cur.store_id.nunique()))
    by = cur.groupby("store_type")[["sales", "forecast"]].sum()
    for t, r in by.iterrows():
        fp.add(f"Store type {t} sales vs forecast", float(r.sales / r.forecast - 1),
               pct(r.sales / r.forecast - 1, 1))
    fp.add("Forecast accuracy (store-week WAPE, backtest)", s.bt["weekly_store_wape"]["p50"],
           f"{s.bt['weekly_store_wape']['p50'] * 100:.1f}%")
    k = s.kpis[(s.kpis.segment == "ALL") & (s.kpis.week_start == last)]
    if len(k):
        k = k.iloc[0]
        for col, label in [
            ("sales_vs_forecast_4wk", "Sales vs forecast, rolling 4 weeks"),
            ("yoy_growth", "Same-store sales vs same days last year"),
            ("yoy_customers_growth", "Same-store customer visits vs last year"),
            ("yoy_basket_growth", "Same-store average basket vs last year"),
            ("gap_concentration_top10", "Share of forecast gap from the 10 biggest-gap stores"),
        ]:
            if pd.notna(k[col]):
                fp.add(label, float(k[col]), pct(k[col], 1))
        fp.add("This is a promo week", bool(k.promo_week), "yes" if k.promo_week else "no")
        fp.context["kpi_note"] = ("Week-over-week swings mostly reflect alternating promo weeks; "
                                  "prefer vs-forecast and vs-last-year comparisons.")
    fp.context["note"] = ("Week may be partial (data ends Friday 2015-07-31); compare vs "
                          "forecast, not raw week-over-week, when judging performance.")
    return fp


def episode(s: Store, store_id: int, rec_type: str) -> FactPack:
    e = s.erca[s.erca.store_id == store_id].sort_values("end").iloc[-1]
    fp = FactPack(f"Store {store_id}", rec_type)
    fp.add("Store ID", int(store_id))
    fp.add("Store type", str(e.store_type))
    fp.add("Gap vs forecast", float(e.gap_pct), pct(e.gap_pct))
    fp.add("Gap vs forecast (EUR, whole episode)", float(e.gap), eur(e.gap))
    fp.add("Gap per week (EUR)", float(e.gap / e.weeks), eur(e.gap / e.weeks))
    fp.add("Consecutive anomalous weeks", int(e.weeks))
    fp.add("Episode start", str(e.start.date()))
    fp.add("Episode end", str(e.end.date()))
    fp.add("Main driver (traffic vs basket)", str(e.tb_driver))
    fp.add("Share of gap explained by customer traffic", float(e.tb_traffic_share),
           f"{e.tb_traffic_share * 100:.0f}%")
    fp.add("Customer traffic vs forecast",
           float(e.customers / e.forecast_customers - 1),
           pct(e.customers / e.forecast_customers - 1))
    fp.add("Root-cause classification", str(e.root_cause).replace("_", " "))
    for ev in json.loads(e.evidence):
        fp.add(f"Evidence: {ev['type'].replace('_', ' ')}", ev.get("date", "yes"), ev["detail"])
    if store_id in s.lift.index:
        L = s.lift.loc[store_id]
        fp.add("Store's causal promo lift", float(L.lift), pct(L.lift))
        fp.add("Chain median promo lift", float(s.lift.lift.median()), pct(s.lift.lift.median()))
    from sic.playbooks import library as L

    books = L.for_episode(e, k=1)
    if books:
        b = books[0]
        fp.add("Approved playbook", b["citation"], f"{b['citation']}: {b['title']}")
        if "not" not in b["expected_impact"].lower()[:12]:
            fp.add("Playbook expected impact (causal analysis)", "text", b["expected_impact"])
        for i, play in enumerate(b["plays"][:3], start=1):
            fp.add(f"Playbook play {i}", "text", play)
        fp.context["playbook_rule"] = (f"The action must be one of the approved playbook plays "
                                       f"and must end with the citation ({b['citation']}).")
        fp.context["playbook_citation"] = b["citation"]
    fp.context["confidence"] = "higher" if e.weeks >= 3 else "early signal (1-2 weeks)"
    return fp


def promo_pilot(s: Store) -> FactPack:
    r = s.recs[s.recs.type == "promo_targeting_pilot"].iloc[0]
    ev = json.loads(r.evidence)
    low = s.lift.loc[ev["stores"]] if ev["stores"] else s.lift.head(0)
    n = int(r.action.split(" the ")[1].split(" ")[0])
    fp = FactPack("Promo mechanics pilot", "promo_targeting_pilot")
    fp.add("Chain pooled causal promo lift", s.pl["pooled_lift"], pct(s.pl["pooled_lift"], 1))
    fp.add("Stores in lowest-lift group", n)
    fp.add("Median lift in that group", float(s.lift.lift.nsmallest(n).median()),
           pct(s.lift.lift.nsmallest(n).median()))
    fp.add("Chain median store lift", s.pl["median_store_lift"], pct(s.pl["median_store_lift"]))
    fp.add("Share of group that is store type b or 'extra' assortment",
           ev["share_type_b_or_extra"], f"{ev['share_type_b_or_extra'] * 100:.0f}%")
    fp.add("Weekly promo revenue at stake in the group", ev["weekly_promo_revenue_at_stake"],
           eur(ev["weekly_promo_revenue_at_stake"]))
    for t, v in s.pl["by_store_type"].items():
        fp.add(f"Median promo lift, store type {t}", v, pct(v))
    fp.add("Proposed test design", "randomize half the group to an alternative mechanic "
           "for 2 promo cycles")
    fp.context["caveats"] = ev["caveat"]
    fp.context["example_stores"] = list(low.index[:5])
    return fp


def promo2(s: Store) -> FactPack:
    d = s.did
    fp = FactPack("Promo2 program effect", "promo2_program")
    fp.add("Estimated effect of enrolling (DiD)", d["effect"]["effect_pct"],
           pct(d["effect"]["effect_pct"], 1))
    fp.add("95% CI low", d["effect"]["ci_pct"][0], pct(d["effect"]["ci_pct"][0], 1))
    fp.add("95% CI high", d["effect"]["ci_pct"][1], pct(d["effect"]["ci_pct"][1], 1))
    fp.add("Trend-adjusted estimate", d["robustness"]["trend_adjusted"]["effect_pct"],
           pct(d["robustness"]["trend_adjusted"]["effect_pct"], 1))
    fp.add("Donut estimate (drop 12 weeks before enrollment)",
           d["robustness"]["donut_drop_12w_pre"]["effect_pct"],
           pct(d["robustness"]["donut_drop_12w_pre"]["effect_pct"], 1))
    fp.add("Placebo estimate (should be ~0)", d["placebo"]["effect_pct"],
           pct(d["placebo"]["effect_pct"], 1))
    fp.add("Pre-trend test p-value", d["pretrend_test"]["p_value"],
           f"{d['pretrend_test']['p_value']:.4f}")
    fp.add("Newly enrolled stores analysed", d["treated_stores"])
    fp.add("Never-enrolled comparison stores", d["control_stores"])
    fp.context["interpretation"] = (
        "Positive in every specification, but enrolled stores were declining before "
        "enrollment and the placebo is non-zero, so part of the estimate may be rebound. "
        "Must NOT be described as proven or definitive.")
    return fp


def segment_anomaly(s: Store) -> FactPack | None:
    if not len(s.segments):
        return None
    r = s.segments.sort_values("robust_z", ascending=False).iloc[0]
    fp = FactPack(f"Store type {r.store_type}, week of {r.week_start:%Y-%m-%d}", "segment_anomaly")
    fp.add("Segment sales vs forecast", float(r.gap_pct), pct(r.gap_pct, 1))
    fp.add("Share of stores moving the same way", float(r.breadth_same_direction),
           f"{r.breadth_same_direction * 100:.0f}%")
    fp.add("Root cause", r.root_cause["label"].replace("_", " ")
           if isinstance(r.root_cause, dict) else str(r.root_cause))
    rc = r.root_cause if isinstance(r.root_cause, dict) else {}
    if rc.get("detail"):
        fp.add("Root-cause detail", "text", rc["detail"])
    for day, g in json.loads(r.by_day).items():
        if g is not None:
            fp.add(f"Sales vs forecast on {day}", g, pct(g))
    return fp
