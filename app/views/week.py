"""Week: how the chain did, and the latest insight briefing."""

from __future__ import annotations

import re
import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACCENT, MUTED, chart, eur, load, pct, require, week_label

from sic import config

_REFS = re.compile(r"\s*\[f\d+(?:\s*,\s*f\d+)*\]")  # fact ids the fact-checker uses

kpis = load("weekly_kpis.parquet")
require(kpis)

# ---------------------------------------------------------------- controls
SEGMENTS = {"ALL": "Chain", "a": "Type a", "b": "Type b", "c": "Type c", "d": "Type d"}
c1, c2, _ = st.columns([1, 1.4, 3])
seg = c1.selectbox("Segment", list(SEGMENTS), format_func=SEGMENTS.get,
                   label_visibility="collapsed")
k = kpis[kpis.segment == seg].sort_values("week_start").reset_index(drop=True)
labels = {r.week_start: week_label(r.week_start, int(r.days_in_data)) for r in k.itertuples()}
wk = c2.selectbox("Week", list(labels)[::-1], format_func=labels.get,
                  label_visibility="collapsed")
cur = k[k.week_start == wk].iloc[0]

# ---------------------------------------------------------------- numbers
m = st.columns(4)
m[0].metric("Sales", eur(cur.sales))
m[1].metric("vs forecast", pct(cur.sales_vs_forecast))
m[2].metric("vs last year", pct(cur.yoy_growth))
m[3].metric("Anomalous stores", f"{cur.stores_below_forecast + cur.stores_above_forecast:.0f}",
            f"{cur.stores_below_forecast:.0f} below", delta_color="off", delta_arrow="off")

fig = go.Figure()
fig.add_scatter(x=k.week_start, y=k.sales_vs_forecast, name="Weekly",
                line={"color": MUTED, "width": 1.5})
fig.add_scatter(x=k.week_start, y=k.sales_vs_forecast_4wk, name="4-week",
                line={"color": ACCENT, "width": 2.5})
fig.add_vline(x=wk, line_dash="dot", line_color=MUTED)
chart(fig, height=240, y="pct", title="Sales vs forecast")

if seg == "ALL":
    t = kpis[(kpis.week_start == wk) & (kpis.segment != "ALL")]
    st.dataframe(pd.DataFrame({
        "Type": t.segment, "Sales": t.sales.map(eur), "vs forecast": t.sales_vs_forecast.map(pct),
        "vs last year": t.yoy_growth.map(pct), "Basket": t.basket.map(lambda v: f"€{v:.2f}"),
        "Anomaly rate": t.anomaly_rate.map(lambda v: pct(v, sign=False))}),
        hide_index=True, width="stretch")


# ---------------------------------------------------------------- insights
def live_cards() -> tuple[list | None, str | None]:
    """Generate cards with the live LLM inside a time budget; None + reason on failure or a
    degraded run (the saved cards stay on screen)."""
    from sic.insights import cards as C
    from sic.insights.demo import health_check

    cfg = config.load()["insights"]
    reason = health_check(cfg.get("health_check_timeout_s", 10))
    if reason:
        return None, reason
    res: dict = {}

    def work():
        try:
            res["cards"] = C.run()
        except Exception as e:  # noqa: BLE001 - any provider failure falls back to saved cards
            res["error"] = f"{type(e).__name__}: {str(e)[:100]}"

    budget = cfg.get("live_time_budget_s", 240)
    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(budget)
    if th.is_alive():
        return None, f"took longer than {budget}s"
    if "error" in res:
        return None, res["error"]
    rate = sum(c["qa"]["published"] for c in res["cards"]) / max(len(res["cards"]), 1)
    if rate < cfg.get("snapshot_min_publish_rate", 0.8):
        return None, f"only {rate:.0%} passed QA"
    return res["cards"], None


h1, h2 = st.columns([6, 1], vertical_alignment="bottom")
h1.subheader("Insights")
if h2.button("Refresh", type="tertiary", icon=":material/refresh:", width="stretch"):
    with st.spinner("Writing cards…"):
        fresh, reason = live_cards()
    if fresh:
        st.session_state["cards"] = fresh
        st.toast("Cards refreshed")
    else:
        st.toast(f"Live cards unavailable ({reason}). Showing saved cards.")

if "cards" in st.session_state:
    cards = st.session_state["cards"]
else:
    from sic.insights.demo import load_snapshot

    snap = load_snapshot()
    cards = snap["cards"] if snap else []
cards = [c for c in cards if c["qa"]["published"]]
if not cards:
    st.write("No cards yet. Press Refresh or run `make cards`.")

cols = st.columns(2)
for i, c in enumerate(cards):
    card = c["card"]
    with cols[i % 2].container(border=True):
        st.markdown(f"**{_REFS.sub('', card.get('title') or c['subject'])}**")
        st.markdown(_REFS.sub("", card.get("headline", "")))
        bullets = [_REFS.sub("", b) for b in card.get("bullets", [])]
        if bullets:
            st.markdown("\n".join(f"- {b}" for b in bullets))
        if card.get("action"):
            st.markdown(f"→ **{_REFS.sub('', card['action'])}**")
