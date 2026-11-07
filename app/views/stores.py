"""Stores: pick a store; see what it's worth, why, its competition, and what to do."""

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
    load_json,
    pct,
    require,
    simulation,
    stores,
)

from sic.playbooks import library as PL

opp, sw, ep = load("opportunity_scores.parquet"), load("store_weeks.parquet"), load(
    "episode_rca.parquet")
require(opp, sw, ep)
recs, comp_fx = load("recommendations.parquet"), load("competitor_store_effects.parquet")
comp = load_json("m3_competitor_impact.json")
dim = stores()

# ---------------------------------------------------------------- pick a store
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

# ---------------------------------------------------------------- summary
st.subheader(f"Store {sid}")
m = st.columns(4)
m[0].metric("Upside / week", eur(r.score_eur))
m[1].metric("Promo lift", pct(r.lift, 0))
m[2].metric("vs forecast, 13 wk", pct(r.gap_recent))
dist = dim.competition_distance_m.get(sid)
m[3].metric("Nearest competitor", "None recorded" if pd.isna(dist) else f"{dist:,.0f} m")
st.write(r.why[:1].upper() + r.why[1:] + ".")
if recs is not None:
    for x in recs[recs.store_id == sid].itertuples():
        st.markdown(f"→ **{x.action}**")

s = sw[(sw.store_id == sid) & sw.week_start.isin(complete_weeks())].sort_values("week_start")
e = ep[ep.store_id == sid].sort_values("start", ascending=False).head(5)
pb = PL.for_store(sid)["playbooks"]
names = ["Sales", "What-if", "Competition", "Similar stores"] + (
    [f"Anomalies · {len(e)}"] if len(e) else []) + (["Playbook"] if pb else [])
tabs = dict(zip(names, st.tabs(names)))

# ---------------------------------------------------------------- tabs
with tabs["Sales"]:
    a = s[s.is_anomaly]
    fig = go.Figure()
    fig.add_scatter(x=s.week_start, y=s.forecast, name="Forecast", line={"color": MUTED})
    fig.add_scatter(x=s.week_start, y=s.sales, name="Actual", line={"color": ACCENT, "width": 2})
    fig.add_scatter(x=a.week_start, y=a.sales, mode="markers", name="Anomaly",
                    marker={"color": SECOND, "size": 8})
    chart(fig, height=260, y="eur", key=f"sales_{sid}")

with tabs["What-if"]:
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

with tabs["Competition"]:
    if pd.isna(dist):
        st.write("No competitor recorded for this store.")
    else:
        all_d = dim.competition_distance_m.dropna()
        opened = dim.competition_open_date.get(sid)
        band = comp["by_distance"]["within_1000m" if dist < 1000 else "beyond_1000m"] \
            if comp else None
        own = comp_fx[comp_fx.store_id == sid] if comp_fx is not None else pd.DataFrame()
        k = st.columns(4)
        k[0].metric("Closer than", f"{(all_d > dist).mean():.0%} of stores")
        k[1].metric("Opened", "Unknown" if pd.isna(opened) else f"{pd.Timestamp(opened):%b %Y}")
        if band:
            k[2].metric(f"Typical impact {'< 1 km' if dist < 1000 else '≥ 1 km'}",
                        pct(band["effect_pct"]),
                        f"{pct(band['ci_pct'][0])} to {pct(band['ci_pct'][1])}",
                        delta_color="off", delta_arrow="off")
        if len(own):
            o = own.iloc[0]
            k[3].metric("This store after opening", pct(o.effect_pct),
                        f"{pct(o.ci_lo_pct)} to {pct(o.ci_hi_pct)}", delta_color="off",
                        delta_arrow="off")

with tabs["Similar stores"]:
    ids = [sid, *[int(i) for i in r.peers]]
    p = opp.set_index("store_id").loc[ids]
    st.dataframe(pd.DataFrame({
        "Store": [f"{i} (selected)" if i == sid else str(i) for i in ids],
        "Sales / day": p.spd.map(eur).values, "Basket": p.basket.map(lambda x: f"€{x:.2f}").values,
        "vs forecast, 13 wk": p.gap_recent.map(pct).values,
        "Competitor": [("—" if pd.isna(x) else f"{x:,.0f} m")
                       for x in dim.competition_distance_m.reindex(ids)],
        "Trend": p.momentum.str.capitalize().values}), hide_index=True, width="stretch")

if len(e):
    with tabs[names[4]]:
        for x in e.itertuples():
            causes = "; ".join(d["detail"] for d in json.loads(x.evidence))
            st.markdown(f"**{x.start:%-d %b %Y}** · {pct(x.gap_pct, 0)} · "
                        f"{x.root_cause.replace('_', ' ')}  \n"
                        f"{eur(x.gap)} over {x.weeks} weeks, {x.tb_driver}-driven. {causes}")

if pb:
    b = pb[0]
    with tabs["Playbook"]:
        st.markdown(f"**{b['title']}** · {b['citation']}")
        st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(b["plays"], 1)))
        if b["do_not"]:
            st.markdown("**Don't** " + "; ".join(b["do_not"]))
        st.markdown("**Measure** " + "; ".join(b["measure"]))
