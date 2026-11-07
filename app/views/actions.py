"""Actions: options to grow revenue, chain-wide and store by store, with the evidence."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (
    ACCENT,
    MUTED,
    SECOND,
    chart,
    complete_weeks,
    eur,
    load,
    pct,
    require,
    simulation,
    stores,
)

opp, recs, sw, ep = (load("opportunity_scores.parquet"), load("recommendations.parquet"),
                     load("store_weeks.parquet"), load("episode_rca.parquet"))
require(opp, recs, sw, ep)

view = st.segmented_control("View", ["Chain", "Stores"], default="Chain", required=True,
                            label_visibility="collapsed")

# ---------------------------------------------------------------- chain programmes
if view == "Chain":
    top = opp.head(10).score_eur.sum()
    chain = recs[recs.store_id.isna()].sort_values("rank")
    m = st.columns(3)
    m[0].metric("Chain programmes", len(chain))
    m[1].metric("Store actions", int(recs.store_id.notna().sum()))
    m[2].metric("Top 10 stores, upside / week", eur(top))
    COLOR = {"high": "green", "medium": "blue", "low": "orange"}
    for x in chain.itertuples():
        ev = json.loads(x.evidence) if isinstance(x.evidence, str) else {}
        with st.container(border=True):
            st.markdown(f"**{x.action}**")
            tags = f":{COLOR.get(x.confidence, 'gray')}-badge[{x.confidence.capitalize()} confidence]"
            if pd.notna(x.weekly_eur_impact):
                tags += f" :gray-badge[{eur(x.weekly_eur_impact)} revenue / week]"
            st.markdown(tags)
            note = ev.get("reason") or ev.get("assumption") or ev.get("caveat")
            if note:
                st.write(note)

# ---------------------------------------------------------------- one store
else:
    dim = stores()
    LEVERS = {"all": "All", "promo": "Promos", "peer": "Basket", "competitor": "Competitor"}
    c1, c2, _ = st.columns([2, 3, 2], vertical_alignment="center")
    lever = c1.segmented_control("Lever", list(LEVERS), format_func=LEVERS.get, default="all",
                                 required=True, label_visibility="collapsed")
    v = opp if lever == "all" else opp[opp.primary_lever == lever]
    label = dict(zip(v.store_id, [f"{s} · {eur(u)}/wk · {LEVERS[lv]}"
                                  for s, u, lv in zip(v.store_id, v.score_eur, v.primary_lever)]))
    sid = int(c2.selectbox("Store", list(label), format_func=label.get, key=f"store_{lever}",
                           label_visibility="collapsed"))
    r = opp[opp.store_id == sid].iloc[0]

    st.subheader(f"Store {sid}")
    m = st.columns(4)
    m[0].metric("Upside / week", eur(r.score_eur))
    m[1].metric("Promo lift", pct(r.lift, 0))
    m[2].metric("vs forecast, 13 wk", pct(r.gap_recent))
    dist = dim.competition_distance_m.get(sid)
    m[3].metric("Nearest competitor", "None recorded" if pd.isna(dist) else f"{dist:,.0f} m")
    st.write(r.why[:1].upper() + r.why[1:] + ".")
    for x in recs[recs.store_id == sid].itertuples():
        st.markdown(f"→ **{x.action}**")

    s = sw[(sw.store_id == sid) & sw.week_start.isin(complete_weeks())].sort_values("week_start")
    e = ep[ep.store_id == sid].sort_values("start", ascending=False).head(5)
    names = ["Sales", "What-if"] + ([f"Anomalies · {len(e)}"] if len(e) else [])
    tabs = st.tabs(names)

    with tabs[0]:
        a = s[s.is_anomaly]
        fig = go.Figure()
        fig.add_scatter(x=s.week_start, y=s.forecast, name="Forecast", line={"color": MUTED})
        fig.add_scatter(x=s.week_start, y=s.sales, name="Actual",
                        line={"color": ACCENT, "width": 2})
        fig.add_scatter(x=a.week_start, y=a.sales, mode="markers", name="Anomaly",
                        marker={"color": SECOND, "size": 8})
        chart(fig, height=260, y="eur", key=f"sales_{sid}")

    with tabs[1]:
        from sic.whatif.engine import run as run_sim

        q1, q2, _ = st.columns([3, 2, 3], vertical_alignment="bottom")
        scen = q1.segmented_control("Scenario", ["Extra promo week", "New competitor"],
                                    default="Extra promo week", required=True,
                                    label_visibility="collapsed", key=f"scen_{sid}")
        if scen == "Extra promo week":
            simulation(run_sim(scenario="extra_promo_week", store_ids=[sid]), f"promo_{sid}")
        else:
            d = q2.number_input("Distance (m)", 10, 20000, 500, step=100, key=f"dist_{sid}")
            simulation(run_sim(scenario="competitor_opens", store_ids=[sid],
                               competitor_distance_m=float(d)), f"comp_{sid}")

    if len(e):
        with tabs[2]:
            for x in e.itertuples():
                causes = "; ".join(d["detail"] for d in json.loads(x.evidence))
                st.markdown(f"**{x.start:%-d %b %Y}** · {pct(x.gap_pct, 0)} · "
                            f"{x.root_cause.replace('_', ' ')}  \n"
                            f"{eur(x.gap)} over {x.weeks} weeks, {x.tb_driver}-driven. {causes}")
