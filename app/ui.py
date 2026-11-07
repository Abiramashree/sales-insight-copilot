"""Shared UI helpers: data loading, number formatting, charts.

Pages import from here and nothing else in app/; all data comes from the pipeline outputs
(data/outputs, reports/). Loads are cached and busted by file mtime, so re-running the
pipeline (or the live feed loading a week) shows up on the next rerun.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sic import config

ACCENT, SECOND, MUTED = "#2a78d6", "#eb6834", "#a3a29c"


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def load(name: str) -> pd.DataFrame | None:
    p = config.path("outputs") / name     # resolved per call: follows the live feed
    return _parquet(str(p), p.stat().st_mtime) if p.exists() else None


def load_json(name: str) -> dict | None:
    p = config.path("reports") / name
    return _json(str(p), p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner=False)
def _complete_weeks(path: str, mtime: float) -> frozenset:
    f = pd.read_parquet(path, columns=["date", "week_start"]).drop_duplicates()
    days = f.groupby("week_start").date.nunique()
    return frozenset(days[days == 7].index)


def complete_weeks() -> frozenset:
    """Weeks with all 7 calendar days forecast. The first (starts Friday 2014-08-01) and the
    last (ends Friday 2015-07-31) are partial, so their € totals are not comparable."""
    p = config.path("outputs") / "forecasts_calibrated.parquet"   # only weeks loaded so far
    return _complete_weeks(str(p), p.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _stores(path: str, mtime: float) -> pd.DataFrame:
    con = config.connect(read_only=True)
    try:
        return con.execute("select * from marts.dim_store").df().set_index("store_id")
    finally:
        con.close()


def stores() -> pd.DataFrame:
    """Store attributes (type, assortment, nearest competitor distance and opening date)."""
    p = config.path("warehouse")
    return _stores(str(p), p.stat().st_mtime)


def require(*items) -> None:
    """Stop the page with one line if a pipeline output is missing."""
    if any(i is None for i in items):
        st.warning("Pipeline outputs missing. Run `make all`.")
        st.stop()


# ---------------------------------------------------------------- formatting
def eur(x: float) -> str:
    a = abs(x)
    s = f"€{a / 1e6:.2f}M" if a >= 1e6 else f"€{a / 1e3:.1f}k" if a >= 1e4 else f"€{a:,.0f}"
    return ("-" if x < 0 else "") + s


def pct(x, d: int = 1, sign: bool = True) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:{'+' if sign else ''}.{d}f}%"


def week_label(start: pd.Timestamp, days: int = 7) -> str:
    """'20–26 Jul 2015'; a partial week shows its real last day ('27–31 Jul 2015')."""
    end = start + pd.Timedelta(days=days - 1)
    left = f"{start:%-d}" if start.month == end.month else f"{start:%-d %b}"
    return f"{left}–{end:%-d %b %Y}"


# ---------------------------------------------------------------- charts
def chart(fig: go.Figure, height: int = 280, y: str | None = None, title: str | None = None,
          key: str | None = None, container=st) -> None:
    """One look for every chart: no background, faint y grid, legend only for 2+ series,
    title in its own row so it never collides with the legend. y = 'pct' | 'eur' | None."""
    named = [t for t in fig.data if t.name and t.showlegend is not False]
    legend = len(named) > 1
    fig.update_layout(
        height=height, font={"size": 12},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title={"text": title, "x": 0, "xanchor": "left", "y": 1, "yanchor": "top",
               "yref": "container", "pad": {"t": 6}, "font": {"size": 14}} if title else None,
        showlegend=legend,
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom"},
        margin={"l": 0, "r": 0, "t": (34 if title else 4) + (28 if legend else 0), "b": 0},
        hovermode=fig.layout.hovermode or "x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.4)",
                     tickformat="+.0%" if y == "pct" else None,
                     tickprefix="€" if y == "eur" else None)
    container.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


# ---------------------------------------------------------------- what-if result
def simulation(r: dict, key: str) -> None:
    """A what-if result: impact, range, per week, confidence, then the biggest stores.
    Shared by the Stores page, the Simulate form and the chat's sources."""
    t, w, pc = r["total_eur"], r["per_week_eur"], r["pct_of_baseline"]
    m = st.columns(4)
    m[0].metric("Expected impact", eur(t["mean"]), pct(pc["mean"]), delta_color="off")
    m[1].metric("90% range", f"{eur(t['p5'])} to {eur(t['p95'])}")
    m[2].metric("Per week", eur(w["mean"]))
    m[3].metric("Confidence", r["confidence"].capitalize())
    if len(r["top_stores"]) > 1:
        top = r["top_stores"][::-1]
        fig = go.Figure(go.Bar(y=[f"Store {x['store_id']}" for x in top],
                               x=[x["expected_eur"] for x in top], orientation="h",
                               marker_color=ACCENT if t["mean"] >= 0 else SECOND))
        fig.update_xaxes(tickprefix="€")
        chart(fig.update_layout(hovermode="closest"), height=40 + 24 * len(top), key=f"sim_{key}")
    with st.expander("Method and assumptions"):
        st.markdown(f"{r['effect_used']}  \n{r['method']}  \n{r['uncertainty']}")
        st.markdown("\n".join(f"- {a}" for a in r["assumptions"] + r["caveats"]))


# ---------------------------------------------------------------- assistant blocks
TOOLS = {"run_sql": "SQL", "explain_store": "Store explainer", "lookup_definition": "Definitions",
         "get_recommendations": "Recommendations", "get_playbook": "Playbook",
         "simulate_scenario": "Simulation", "get_opportunities": "Opportunities",
         "test_hypothesis": "Hypothesis test", "plan_ab_test": "A/B sizing",
         "plan_initiative": "Plan", "design_ab_test": "A/B design",
         "analyze_ab_test": "A/B readout"}


def unverified(work: dict) -> None:
    if work.get("unsupported_numbers"):
        st.warning("Not found in any source: " + ", ".join(work["unsupported_numbers"]))


def sources(work: dict, key: str) -> None:
    """What an answer was built from: each tool call with its input and result."""
    calls = work.get("tool_calls") or []
    if not calls:
        return
    with st.expander(f"Sources · {len(calls)}"):
        for i, c in enumerate(calls, 1):
            st.markdown(f"**{TOOLS.get(c['tool'], c['tool'])}**" + ("" if c["ok"] else " · failed"))
            try:
                args = {k: v for k, v in json.loads(c["arguments"] or "{}").items()
                        if v is not None}
            except json.JSONDecodeError:
                args = {"raw": c["arguments"]}
            if c["tool"] == "run_sql" and "query" in args:
                st.code(args["query"], language="sql")
            elif args:
                st.code(json.dumps(args, indent=1), language="json")
            try:
                res = json.loads(c.get("result_preview") or "")
            except json.JSONDecodeError:
                res = None
            if isinstance(res, dict) and "rows" in res and "columns" in res:
                st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]).head(20),
                             hide_index=True, width="stretch")
            elif res is not None:
                st.json(res, expanded=False)


def fmt_p(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def readout(r: dict) -> None:
    """An A/B readout: decision, CUPED lift with CI, raw for comparison."""
    best = r.get("cuped") or r["raw"]
    go_ = r["decision"] == "roll out"
    st.markdown((":green-badge[Roll out]" if go_ else ":orange-badge[Do not roll out]")
                + (" :gray-badge[Dry run on history]" if r.get("dry_run") else ""))
    m = st.columns(3)
    m[0].metric("Lift" + (" (CUPED)" if r.get("cuped") else ""), pct(best["lift"]))
    m[1].metric("p-value", fmt_p(best["p_value"]))
    m[2].metric("Stores", f"{r['stores_test']} vs {r['stores_control']}")
    st.markdown(f":gray[95% CI {pct(best['ci'][0])} to {pct(best['ci'][1])}"
                + (f" · raw lift {pct(r['raw']['lift'])}" if r.get("cuped") else "")
                + f" · {r['weeks']} weeks from {r['test_start']}]")
    st.write(r["rule"])


def plan_block(p: dict, key: str) -> None:
    """The deterministic plan behind an assistant answer: stores, alternatives, design."""
    pl, b = p["plausibility"], p["baseline"]
    color = {"Plausible": "green", "Stretch": "blue", "Unlikely": "orange"}[pl["verdict"]]
    st.markdown(f":{color}-badge[{pl['verdict']}] :gray-badge[upside covers "
                f"{pl['coverage']:.2f}× the goal]")
    m = st.columns(3)
    m[0].metric("Weekly sales now", eur(b["weekly_sales_eur"]))
    m[1].metric("Needed per week", eur(b["required_eur_per_week"]),
                pct(p["goal"]["target_pct"], 1), delta_color="off", delta_arrow="off")
    m[2].metric("Covered by plan", eur(p["covered_eur_per_week"]),
                f"gap {eur(p['gap_eur_per_week'])}" if p["gap_eur_per_week"] > 1 else None,
                delta_color="off", delta_arrow="off")
    t1, t2, t3, t4 = st.tabs([f"Stores · {len(p['stores'])}", "Alternatives", "Hypotheses",
                              "Experiment"])
    with t1:
        s = pd.DataFrame(p["stores"])
        if len(s):
            st.dataframe(pd.DataFrame({
                "Store": s.store_id, "Type": s.store_type,
                "Weekly sales": s.weekly_sales.map(eur), "Add / week": s.add_eur_per_week.map(eur),
                "Add %": s.add_pct.map(pct), "Lever": s.main_lever}),
                hide_index=True, width="stretch", height=min(36 * len(s) + 38, 300))
    with t2:
        st.dataframe(pd.DataFrame([{
            "Option": a["option"], "€ / week": eur(a["eur_per_week"]),
            "Range": f"{eur(a['range'][0])} to {eur(a['range'][1])}",
            "Confidence": a["confidence"].capitalize()} for a in p["alternatives"]]),
            hide_index=True, width="stretch")
    with t3:
        for h in p["hypotheses"]:
            st.markdown(f"**{h['id']}** If {h['if']}, then {h['then']}, because {h['because']}.")
            if "effect_pct" in h:
                st.markdown(f"{h['test']}: {pct(h['effect_pct'])} (95% CI {pct(h['ci_pct'][0])}"
                            f" to {pct(h['ci_pct'][1])}, p {fmt_p(h['p_value'])}, "
                            f"{h['n_a']} stores)")
            else:
                st.markdown(f"{h['test']}: customers {pct(h['customers_effect_pct'])}, "
                            f"basket {pct(h['basket_effect_pct'])}: {h['result']}")
    with t4:
        e = p["experiment"] or {}
        if "note" in e:
            st.write(e["note"])
        else:
            st.markdown(":green-badge[Powered]" if e["powered"] else
                        f":orange-badge[Under-powered: needs {e['stores_needed_cuped']} stores "
                        "per group]")
            k = st.columns(3)
            k[0].metric("Stores per group", e["stores_per_arm"])
            k[1].metric("Weeks", e["weeks"])
            k[2].metric("Detectable lift", pct(e["mde_cuped"]), f"target {pct(e['target_lift'])}",
                        delta_color="off", delta_arrow="off")
            st.write(f"Primary metric: {e['primary_metric']}. Guardrail: {e['guardrail']}. "
                     f"Decision: {e['decision_rule']}.")
            g = pd.DataFrame(e["groups"])
            st.dataframe(g.rename(columns={"store_id": "Store", "group": "Group",
                                           "weekly_sales": "Weekly sales"}),
                         hide_index=True, width="stretch", height=200, key=f"groups_{key}")
