"""Pipeline: data arriving, the week's KPIs, its anomalies and why, and the briefing."""

from __future__ import annotations

import json
import re
import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACCENT, MUTED, SECOND, chart, complete_weeks, eur, load, pct, require, week_label

from sic import config, feed
from sic.insights.demo import load_snapshot

# ---------------------------------------------------------------- feed
state = config.feed_state()
f1, f2, f3, f4 = st.columns([3, 1, 1, 1], vertical_alignment="center")
if state:
    f1.markdown(f"**Live** · data through {pd.Timestamp(state['as_of']):%-d %b %Y}")
    at_end = state["as_of"] >= config.load()["feed"]["end"]
    playing = st.session_state.get("playing", False)
    if f2.button("Next week", type="primary", disabled=at_end or playing, width="stretch"):
        with st.spinner("Loading the next week…"):
            feed.next_week()
        st.rerun()
    if f3.button("Pause" if playing else "Play", disabled=at_end, width="stretch"):
        st.session_state["playing"] = not playing
        st.rerun()
    if f4.button("Stop", width="stretch"):
        st.session_state["playing"] = False
        feed.stop()
        st.rerun()
    runs = state.get("runs", [])
    if runs:
        steps = runs[-1]["steps"]
        for col, s in zip(st.columns(len(steps)), steps):
            col.markdown(f"**{s['step']}**  \n{s['result']}  \n:gray[{s['seconds']:.1f}s]")
else:
    f1.markdown(f"**Full history** · data through "
                f"{pd.Timestamp(config.load()['feed']['end']):%-d %b %Y}")
    if f4.button("Go live", type="primary", width="stretch"):
        with st.spinner("Loading history and running every step…"):
            feed.start()
        st.rerun()

kpis = load("weekly_kpis.parquet")
require(kpis)

# ---------------------------------------------------------------- KPIs
SEGMENTS = {"ALL": "Chain", "a": "Type a", "b": "Type b", "c": "Type c", "d": "Type d"}
st.divider()
c1, c2, _ = st.columns([1, 1.4, 3])
seg = c1.selectbox("Segment", list(SEGMENTS), format_func=SEGMENTS.get,
                   label_visibility="collapsed")
k = kpis[kpis.segment == seg].sort_values("week_start").reset_index(drop=True)
labels = {r.week_start: week_label(r.week_start, int(r.days_in_data)) for r in k.itertuples()}
wk = c2.selectbox("Week", list(labels)[::-1], format_func=labels.get,
                  label_visibility="collapsed", key=f"week_{state['as_of'] if state else 'full'}")
cur = k[k.week_start == wk].iloc[0]

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
chart(fig, height=220, y="pct", title="Sales vs forecast")

# ---------------------------------------------------------------- anomalies
sw, ep = load("store_weeks.parquet"), load("episode_rca.parquet")
a = sw[(sw.week_start == wk) & sw.is_anomaly] if sw is not None else pd.DataFrame()
if seg != "ALL" and len(a):
    a = a[a.store_type == seg]
st.subheader(f"Anomalies · {len(a)}")
if len(a):
    e = ep[(ep.start <= wk) & (ep.end >= wk)][["store_id", "root_cause", "tb_driver", "evidence"]]
    a = a.merge(e, on="store_id", how="left").sort_values("gap", key=abs, ascending=False)
    a = a.reset_index(drop=True)
    left, right = st.columns([3, 2], gap="large")
    sel = left.dataframe(pd.DataFrame({
        "Store": a.store_id, "Type": a.store_type, "vs forecast": a.gap_pct.map(pct),
        "Gap": a.gap.map(eur), "Cause": a.root_cause.fillna("—").str.replace("_", " "),
        "Driver": a.tb_driver.fillna("—")}),
        hide_index=True, width="stretch", height=min(36 * len(a) + 38, 320),
        on_select="rerun", selection_mode="single-row", key=f"anom_{wk}_{seg}",
        selection_default={"selection": {"rows": [0]}})
    x = a.iloc[sel.selection.rows[0] if sel.selection.rows else 0]
    with right:
        s = sw[(sw.store_id == x.store_id) & sw.week_start.isin(complete_weeks())]
        s = s.sort_values("week_start").tail(16)
        fig = go.Figure()
        fig.add_scatter(x=s.week_start, y=s.forecast, name="Forecast", line={"color": MUTED})
        fig.add_scatter(x=s.week_start, y=s.sales, name="Actual", line={"color": ACCENT})
        hit = s[s.is_anomaly]
        fig.add_scatter(x=hit.week_start, y=hit.sales, mode="markers", name="Anomaly",
                        marker={"color": SECOND, "size": 8})
        chart(fig, height=200, y="eur", title=f"Store {x.store_id}", key=f"anom_chart_{wk}")
        if isinstance(x.evidence, str):
            st.write("; ".join(d["detail"] for d in json.loads(x.evidence)))


# ---------------------------------------------------------------- briefing
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
        except Exception as e:  # noqa: BLE001 - any provider failure keeps the saved cards
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


_REFS = re.compile(r"\s*\[f\d+(?:\s*,\s*f\d+)*\]")      # fact ids used by the fact-checker
latest = kpis.week_start.max()
h1, h2 = st.columns([6, 1], vertical_alignment="bottom")
h1.subheader("Briefing")
if h2.button("Refresh", type="tertiary", icon=":material/refresh:", width="stretch"):
    with st.spinner("Writing the briefing…"):
        fresh, reason = live_cards()
    if fresh:
        st.session_state["cards"] = (str(latest.date()), fresh)
        st.toast("Briefing refreshed")
    else:
        st.toast(f"Live briefing unavailable ({reason}).")

cached = st.session_state.get("cards")
snap = load_snapshot()
if cached and cached[0] == str(latest.date()):
    cards = cached[1]
elif snap and snap["meta"].get("data_week") == str(latest.date()):
    cards = snap["cards"]
else:
    cards = []
cards = [c for c in cards if c["qa"]["published"]]
if not cards:
    st.write(f"No briefing for the week of {latest:%-d %b} yet. Press Refresh to write one.")
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

# ---------------------------------------------------------------- play
if state and st.session_state.get("playing"):
    with st.spinner("Loading the next week…"):
        more = feed.next_week()
    if more is None:
        st.session_state["playing"] = False
    st.rerun()
