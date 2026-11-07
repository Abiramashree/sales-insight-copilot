"""Stores: which stores to act on, why, and what an action is worth."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACCENT, MUTED, SECOND, chart, complete_weeks, eur, load, pct, require, simulation

opp, sw, ep = load("opportunity_scores.parquet"), load("store_weeks.parquet"), load(
    "episode_rca.parquet")
require(opp, sw, ep)
recs = load("recommendations.parquet")

LEVERS = {"all": "All", "promo": "Promos", "peer": "Basket", "competitor": "Competitor"}
left, right = st.columns([2, 3], gap="large")

# ---------------------------------------------------------------- ranked list
with left:
    lever = st.segmented_control("Lever", list(LEVERS), format_func=LEVERS.get, default="all",
                                 required=True, label_visibility="collapsed")
    v = opp if lever == "all" else opp[opp.primary_lever == lever]
    v = v.reset_index(drop=True)
    sel = st.dataframe(
        pd.DataFrame({"Store": v.store_id, "Upside / wk": v.score_eur.map(eur),
                      "Lever": v.primary_lever.map(LEVERS), "Trend": v.momentum.str.capitalize()}),
        hide_index=True, width="stretch", height=560, key=f"rank_{lever}",
        on_select="rerun", selection_mode="single-row",
        selection_default={"selection": {"rows": [0]}})
    rows = sel.selection.rows
    sid = int(v.store_id.iloc[rows[0] if rows else 0])

# ---------------------------------------------------------------- one store
r = opp[opp.store_id == sid].iloc[0]
with right:
    st.subheader(f"Store {sid}")
    m = st.columns(3)
    m[0].metric("Upside / week", eur(r.score_eur))
    m[1].metric("Promo lift", pct(r.lift, 0))
    m[2].metric("vs forecast, 13 wk", pct(r.gap_recent))
    st.write(r.why[:1].upper() + r.why[1:] + ".")
    if recs is not None:
        for x in recs[recs.store_id == sid].itertuples():
            st.markdown(f"→ **{x.action}**")

    s = sw[(sw.store_id == sid) & sw.week_start.isin(complete_weeks())].sort_values("week_start")
    a = s[s.is_anomaly]
    fig = go.Figure()
    fig.add_scatter(x=s.week_start, y=s.forecast, name="Forecast", line={"color": MUTED})
    fig.add_scatter(x=s.week_start, y=s.sales, name="Actual", line={"color": ACCENT, "width": 2})
    fig.add_scatter(x=a.week_start, y=a.sales, mode="markers", name="Anomaly",
                    marker={"color": SECOND, "size": 8})
    chart(fig, height=240, y="eur", key=f"store_{sid}")

    if st.button("Simulate one extra promo week", key=f"simbtn_{sid}"):
        from sic.whatif.engine import run as run_sim

        st.session_state["store_sim"] = run_sim(scenario="extra_promo_week", store_ids=[sid])
    res = st.session_state.get("store_sim")
    if res and res["inputs"].get("store_ids") == [sid]:
        simulation(res, f"store_{sid}")

    e = ep[ep.store_id == sid].sort_values("start", ascending=False).head(5)
    if len(e):
        st.subheader("Anomalies")
        for x in e.itertuples():
            causes = "; ".join(d["detail"] for d in json.loads(x.evidence))
            with st.expander(f"{x.start:%-d %b %Y} · {pct(x.gap_pct, 0)} · "
                             f"{x.root_cause.replace('_', ' ')}"):
                st.write(f"{eur(x.gap)} over {x.weeks} weeks, {x.tb_driver}-driven. {causes}")

    from sic.playbooks import library as PL

    pb = PL.for_store(sid)["playbooks"]
    if pb:
        b = pb[0]
        st.subheader("Playbook")
        with st.expander(f"{b['title']} · {b['citation']}"):
            st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(b["plays"], 1)))
            if b["do_not"]:
                st.markdown("**Don't** " + "; ".join(b["do_not"]))
            st.markdown("**Measure** " + "; ".join(b["measure"]))
