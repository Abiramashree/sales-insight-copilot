"""KPIs: the chain or one store, for a week, month or quarter, and its biggest anomalies."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACCENT, MUTED, SECOND, chart, complete_weeks, eur, load, pct, require

from sic import config
from sic.kpi.period import kpis, periods

sw, ep, wk_kpis, opp = (load("store_weeks.parquet"), load("episode_rca.parquet"),
                        load("weekly_kpis.parquet"), load("opportunity_scores.parquet"))
require(sw, ep, wk_kpis, opp)
recs = load("recommendations.parquet")


@st.cache_data(show_spinner=False)
def _kpis(start, end, store_id, mtime):
    return kpis(start, end, store_id)


@st.cache_data(show_spinner=False)
def _periods(grain, mtime):
    return periods(grain)


mtime = (config.path("outputs") / "forecasts_calibrated.parquet").stat().st_mtime  # cache key

# ---------------------------------------------------------------- controls
c1, c2, c3, _ = st.columns([2, 2, 2, 1], vertical_alignment="center")
stores = sorted(sw.store_id.unique())
scope = c1.selectbox("Scope", ["chain", *stores], label_visibility="collapsed",
                     format_func=lambda s: "Chain" if s == "chain" else f"Store {s}",
                     key="kpi_scope")
grain = c2.segmented_control("Period", ["week", "month", "quarter"], default="week",
                             required=True, format_func=str.capitalize,
                             label_visibility="collapsed")
plist = _periods(grain, mtime)
idx = c3.selectbox("When", range(len(plist)), format_func=lambda i: plist[i][2],
                   label_visibility="collapsed", key=f"kpi_when_{grain}")
start, end, label = plist[idx]
sid = None if scope == "chain" else int(scope)

# ---------------------------------------------------------------- numbers + trend
k = _kpis(start, end, sid, mtime)
m = st.columns(3)
m[0].metric("Sales", eur(k["sales"]) if k["sales"] else "—")
m[1].metric("vs forecast", pct(k["vs_forecast"]))
m[2].metric("vs last year", pct(k["vs_last_year"]))

fig = go.Figure()
if sid is None:
    c = wk_kpis[wk_kpis.segment == "ALL"].sort_values("week_start")
    fig.add_scatter(x=c.week_start, y=c.sales_vs_forecast, name="Weekly",
                    line={"color": MUTED, "width": 1.5})
    fig.add_scatter(x=c.week_start, y=c.sales_vs_forecast_4wk, name="4-week",
                    line={"color": ACCENT, "width": 2.5})
    y, title = "pct", "Sales vs forecast"
else:
    s = sw[(sw.store_id == sid) & sw.week_start.isin(complete_weeks())].sort_values("week_start")
    fig.add_scatter(x=s.week_start, y=s.forecast, name="Forecast", line={"color": MUTED})
    fig.add_scatter(x=s.week_start, y=s.sales, name="Actual", line={"color": ACCENT, "width": 2})
    y, title = "eur", f"Store {sid}, weekly sales"
fig.add_vrect(x0=start - pd.Timedelta(days=start.weekday()), x1=end, fillcolor=ACCENT,
              opacity=0.08, line_width=0)
chart(fig, height=240, y=y, title=title)

# ---------------------------------------------------------------- anomalies
a = sw[sw.is_anomaly & (sw.week_start >= start - pd.Timedelta(days=start.weekday()))
       & (sw.week_start <= end)]
if sid is not None:
    a = a[a.store_id == sid]
a = a.reindex(a.gap.abs().sort_values(ascending=False).index).head(10).reset_index(drop=True)
st.subheader(f"Anomalies · {len(a)}" if len(a) else "No anomalies")
if not len(a):
    st.stop()

sel = st.dataframe(pd.DataFrame({
    "Store": a.store_id, "Week": a.week_start.dt.strftime("%-d %b %Y"),
    "vs forecast": a.gap_pct.map(pct), "Gap": a.gap.map(eur)}),
    hide_index=True, width="stretch", height=min(36 * len(a) + 38, 400),
    on_select="rerun", selection_mode="single-row", key=f"anom_{scope}_{grain}_{idx}",
    selection_default={"selection": {"rows": [0]}})
x = a.iloc[sel.selection.rows[0] if sel.selection.rows else 0]
store, week = int(x.store_id), x.week_start
e = ep[(ep.store_id == store) & (ep.start <= week) & (ep.end >= week)]
e = e.iloc[0] if len(e) else None

st.subheader(f"Store {store} · week of {week:%-d %b %Y}")
why, ctx = st.columns(2, gap="large")

with why:
    st.markdown("**Why**")
    st.metric("Gap vs forecast", eur(x.gap), pct(x.gap_pct), delta_color="off",
              delta_arrow="off")
    if e is not None:
        up = x.gap > 0
        driver = (("more" if up else "fewer") + " customers" if e.tb_driver == "traffic"
                  else ("bigger" if up else "smaller") + " baskets")
        share = e.tb_traffic_share if e.tb_driver == "traffic" else 1 - e.tb_traffic_share
        st.markdown(f"Driven by **{driver}**"
                    + (f" ({share:.0%} of the gap)" if pd.notna(share) else "") + ".  \n"
                    f"Likely cause: **{e.root_cause.replace('_', ' ')}**.")
        details = [d["detail"] for d in json.loads(e.evidence)]
        if details:
            st.markdown("\n".join(f"- {d}" for d in details))

with ctx:
    st.markdown("**Context**")
    # % vs forecast keeps partial weeks (e.g. the last, Mon-Fri) comparable
    s = sw[(sw.store_id == store)
           & sw.week_start.between(week - pd.Timedelta(weeks=8), week + pd.Timedelta(weeks=4))]
    s = s.sort_values("week_start")
    fig = go.Figure()
    fig.add_scatter(x=s.week_start, y=s.gap_pct, name="vs forecast",
                    line={"color": ACCENT, "width": 2})
    hit = s[s.week_start == week]
    fig.add_scatter(x=hit.week_start, y=hit.gap_pct, mode="markers", name="This week",
                    marker={"color": SECOND, "size": 10})
    chart(fig, height=200, y="pct", title="vs forecast, weeks around it",
          key=f"ctx_{store}_{week:%Y%m%d}")
    peers = opp.loc[opp.store_id == store, "peers"]
    peer_ids = [int(i) for i in peers.iloc[0]] if len(peers) else []
    pw = sw[sw.store_id.isin(peer_ids) & (sw.week_start == week)]
    if len(pw):
        med = pw.gap_pct.median()
        moved = (pw.gap_pct * x.gap_pct > 0).mean()
        st.metric("Look-alike stores, same week", pct(med),
                  f"{moved:.0%} moved the same way · this store {pct(x.gap_pct)}",
                  delta_color="off", delta_arrow="off")

st.markdown("**Next step**")
actions = recs[recs.store_id == store] if recs is not None else pd.DataFrame()
for r in actions.itertuples():
    st.markdown(f"→ **{r.action}**")
if actions.empty:
    st.write("No recommended action from the pipeline for this store.")
if st.button("Investigate with the assistant", icon=":material/arrow_forward:"):
    st.session_state["pending"] = f"Investigate store {store}, week of {week:%-d %b %Y}."
    st.switch_page("views/assistant.py")
