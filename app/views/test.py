"""Test: check a hypothesis against the data, or plan an A/B test (no LLM)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ui import ACCENT, MUTED, chart, pct

from sic import hypothesis as H

mode = st.segmented_control("Mode", ["Test", "Plan"], default="Test", required=True,
                            label_visibility="collapsed")

# ---------------------------------------------------------------- test
if mode == "Test":
    COMPARE = {**{k: f"{a} vs {b}" for k, (a, b) in H.WITHIN.items()}, **H.BETWEEN}
    c1, c2, c3, c4 = st.columns(4)
    metric = c1.selectbox("Metric", list(H.METRICS), format_func=H.METRICS.get)
    comp = c2.selectbox("Compare", list(COMPARE), format_func=COMPARE.get)
    a = b = None
    if comp in H.BETWEEN:
        opts = H.groups(comp)
        a = c3.selectbox("Group", opts, index=0, key=f"a_{comp}")
        b = c4.selectbox("Against", [o for o in opts if o != a], key=f"b_{comp}_{a}")
    else:
        stype = c3.selectbox("Stores", ["All", "a", "b", "c", "d"],
                             format_func=lambda t: t if t == "All" else f"Type {t}")
    try:
        r = H.test(metric, comp, a, b,
                   store_type=None if comp in H.BETWEEN or stype == "All" else stype)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    within = r["design"] == "within-store"
    sig = r["p_value"] < 0.05
    st.subheader(f"{r['a']} vs {r['b']}")
    st.markdown((":green-badge[Within-store]" if within else ":orange-badge[Association only]")
                + (" :blue-badge[Significant]" if sig else " :gray-badge[Not significant]"))
    m = st.columns(3)
    m[0].metric("Difference", pct(r["effect_pct"]),
                f"95% CI {pct(r['ci_pct'][0])} to {pct(r['ci_pct'][1])}",
                delta_color="off", delta_arrow="off")
    def fmt_p(x: float) -> str:
        return "< 0.001" if x < 0.001 else f"{x:.3f}"

    m[1].metric("p-value", fmt_p(r["p_value"]), f"rank test {fmt_p(r['p_rank'])}",
                delta_color="off", delta_arrow="off")
    m[2].metric("Stores", f"{r['n_a']:,}" if within else f"{r['n_a']:,} vs {r['n_b']:,}")

    ps = r["per_store"]
    fig = go.Figure()
    if within:
        fig.add_histogram(x=ps.effect, nbinsx=50, marker_color=ACCENT, name="Stores")
        fig.add_vline(x=0, line_color=MUTED)
        fig.update_xaxes(tickformat="+.0%")
        title = "Difference in each store"
    else:
        for g, color in ((r["a"], ACCENT), (r["b"], MUTED)):
            fig.add_box(y=ps[ps.group == g].value, name=str(g), marker_color=color,
                        boxpoints="outliers")
        fig.update_yaxes(tickprefix="€" if metric != "customers" else None)
        title = f"{H.METRICS[metric]} by store"
    chart(fig.update_layout(hovermode="closest", bargap=0.05), height=260, title=title)

# ---------------------------------------------------------------- plan
else:
    c1, c2, c3 = st.columns(3)
    target = c1.number_input("Lift to detect (%)", 0.5, 50.0, 3.0, step=0.5) / 100
    weeks = c2.slider("Test length (weeks)", 2, 12, 4)
    n = c3.number_input("Stores per group", 5, 550, 50, step=5)
    sp = H.weekly_spread(weeks)
    p = H.plan(weeks=weeks, stores_per_arm=int(n), target_lift=target, spread=sp)
    half = p["available_stores"] // 2
    m = st.columns(4)
    m[0].metric("Detectable lift", pct(p["mde_cuped"]), f"{pct(p['mde_raw'])} without CUPED",
                delta_color="off", delta_arrow="off")
    m[1].metric(f"Stores needed for {pct(target, 1, sign=False)}", f"{p['stores_needed_cuped']:,}",
                f"{p['stores_needed_raw']:,} without CUPED", delta_color="off", delta_arrow="off")
    m[2].metric("Significance", f"{p['alpha']:.0%}")
    m[3].metric("Power", f"{p['power']:.0%}")
    ok = p["stores_needed_cuped"] <= min(int(n), half)
    st.badge(f"{int(n)} stores per group is enough" if ok else
             f"Needs {p['stores_needed_cuped']:,} stores per group"
             + ("" if p["stores_needed_cuped"] <= half else f" (only {half} available)"),
             color="green" if ok else "orange")

    xs = np.arange(5, half + 1, 5)
    fig = go.Figure()
    for k, color, name in (("raw", MUTED, "Without CUPED"), ("cuped", ACCENT, "With CUPED")):
        fig.add_scatter(x=xs, y=H.mde(sp[f"sd_{k}"], xs), name=name,
                        line={"color": color, "width": 2.5})
    fig.add_hline(y=target, line_dash="dot", line_color=MUTED)
    fig.update_xaxes(title="Stores per group")
    chart(fig.update_layout(hovermode="x unified"), height=260, y="pct",
          title="Detectable lift by test size")
