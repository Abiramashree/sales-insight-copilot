"""Sales Insight Copilot — Streamlit front end.

  make app        ->  http://localhost:8501

Reads pipeline outputs from data/outputs and reports/; the LLM features (live cards,
chat) fall back to the saved demo snapshot when the provider is down or slow.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sic import config

BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
OUT, REP = config.path("outputs"), config.path("reports")
_STRIP = re.compile(r"\s*\[f\d+(?:\s*,\s*f\d+)*\]")

st.set_page_config(page_title="Sales Insight Copilot", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data
def _mtime(p) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(open(path).read())


def load(name: str) -> pd.DataFrame | None:
    p = OUT / name
    return _read_parquet(str(p), _mtime(p)) if p.exists() else None


def load_json(name: str) -> dict | None:
    p = REP / name
    return _read_json(str(p), _mtime(p)) if p.exists() else None


def eur(x: float) -> str:
    a = abs(x)
    s = f"€{a / 1e6:.2f}M" if a >= 1e6 else f"€{a / 1e3:.1f}k" if a >= 1e4 else f"€{a:,.0f}"
    return ("-" if x < 0 else "") + s


def pct(x, d=1, sign=True) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:{'+' if sign else ''}.{d}f}%"


def style(fig: go.Figure, height: int = 320, pct_axis: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 8, "r": 8, "t": 36, "b": 8},
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend={"orientation": "h", "y": 1.12, "x": 0}, hovermode="x unified",
                      font={"size": 12})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=True,
                     zerolinecolor="rgba(128,128,128,0.5)",
                     tickformat="+.0%" if pct_axis else None)
    return fig


def finalize(fig: go.Figure) -> go.Figure:
    """Title in its own row at the top, legend in a row just above the plot area, top
    margin sized to fit, so titles and legends never overlap."""
    title = fig.layout.title.text if fig.layout.title else None
    named = [t for t in fig.data if t.showlegend is not False and t.name]
    legend = len(named) > 1
    top = (40 if title else 8) + (32 if legend else 0)
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "y": 1, "yanchor": "top",
               "yref": "container", "pad": {"t": 8}, "font": {"size": 14}} if title else None,
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom",
                "font": {"size": 11}},
        showlegend=legend, margin={"l": 8, "r": 8, "t": top, "b": 8})
    return fig


def plot(container, fig: go.Figure, **kw) -> None:
    container.plotly_chart(finalize(fig), **kw)


# ---------------------------------------------------------------- header
st.title("Sales Insight Copilot")
st.caption("Detect what changed, diagnose why, recommend what to do — every number traceable. "
           "Rossmann data (1,115 stores, through 2015-07-31).")

kpis = load("weekly_kpis.parquet")
tabs = st.tabs(["📊 Weekly KPIs", "🗂 Insight cards", "🎯 Opportunities", "🏪 Store explorer",
                "💬 Ask the Copilot", "🔬 Model & methods", "✅ Evals & monitoring"])

# ================================================================ 1. KPIs
with tabs[0]:
    if kpis is None:
        st.warning("Run `make kpis` first.")
    else:
        segs = ["ALL", *sorted(s for s in kpis.segment.unique() if s != "ALL")]
        c1, c2 = st.columns([1, 3])
        seg = c1.selectbox("Segment", segs, format_func=lambda s: "Whole chain" if s == "ALL"
                           else f"Store type {s}")
        k = kpis[kpis.segment == seg].sort_values("week_start").reset_index(drop=True)
        weeks = list(k.week_start.dt.date)
        wk = c2.select_slider("Week", options=weeks, value=weeks[-1])
        i = weeks.index(wk)
        cur, prev = k.iloc[i], k.iloc[i - 1] if i > 0 else None
        if cur.partial_week:
            st.info("Partial week (data ends Friday 2015-07-31). WoW and YoY compare the same "
                    "weekdays, so they stay like-for-like.")
        if cur.promo_week != (prev.promo_week if prev is not None else cur.promo_week):
            st.caption(f"Promo week: **{'yes' if cur.promo_week else 'no'}** vs prior week "
                       f"{'yes' if prev.promo_week else 'no'}: week-over-week swings mostly "
                       "reflect the alternating promo schedule.")

        def delta(col, kind="pct"):
            if prev is None or pd.isna(prev[col]) or pd.isna(cur[col]):
                return None
            d = cur[col] - prev[col]
            return f"{d * 100:+.1f} pp vs prior wk" if kind == "pct" else f"{d / prev[col]:+.1%} vs prior wk"

        m = st.columns(4)
        m[0].metric("Sales", eur(cur.sales), delta("sales", "rel"))
        m[1].metric("Sales vs forecast", pct(cur.sales_vs_forecast), delta("sales_vs_forecast"))
        m[2].metric("YoY same-store", pct(cur.yoy_growth), delta("yoy_growth"))
        m[3].metric("Rolling 4-wk vs forecast", pct(cur.sales_vs_forecast_4wk))
        m = st.columns(4)
        m[0].metric("Customers YoY", pct(cur.yoy_customers_growth))
        m[1].metric("Basket YoY", pct(cur.yoy_basket_growth))
        m[2].metric("Avg basket", f"€{cur.basket:.2f}", delta("basket", "rel"))
        m[3].metric("Sales / open store-day", eur(cur.sales_per_store_day),
                    delta("sales_per_store_day", "rel"))
        m = st.columns(4)
        m[0].metric("Anomaly rate", pct(cur.anomaly_rate, sign=False),
                    f"{cur.stores_below_forecast:.0f} below · {cur.stores_above_forecast:.0f} above",
                    delta_color="off")
        m[1].metric("Gap from top-10 stores", pct(cur.gap_concentration_top10, 0, sign=False),
                    help="High = a few stores drive the week; low = broad movement.")
        m[2].metric("Forecast error (WAPE)", pct(cur.forecast_wape, sign=False),
                    delta("forecast_wape"), delta_color="inverse")
        m[3].metric("Promo sales share", pct(cur.promo_sales_share, 0, sign=False))

        g1, g2 = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=k.week_start, y=k.sales_vs_forecast, name="Weekly",
                        line={"color": NEUTRAL, "width": 1.5})
        fig.add_scatter(x=k.week_start, y=k.sales_vs_forecast_4wk, name="Rolling 4 weeks",
                        line={"color": BLUE, "width": 2.5})
        fig.add_vline(x=pd.Timestamp(wk), line_dash="dot", line_color=NEUTRAL)
        plot(g1, style(fig, pct_axis=True).update_layout(title="Sales vs forecast"),
                        width="stretch")
        ky = k.dropna(subset=["yoy_growth"])
        ky = ky[~ky.partial_week]
        fig = go.Figure()
        fig.add_bar(x=ky.week_start, y=ky.yoy_customers_growth, name="Customers", marker_color=BLUE)
        fig.add_bar(x=ky.week_start, y=ky.yoy_basket_growth, name="Basket", marker_color=ORANGE)
        plot(g2, style(fig, pct_axis=True).update_layout(
            title="Same-store YoY: customers vs basket", barmode="group", bargap=0.25),
            width="stretch")

        if seg == "ALL":
            st.subheader("By store type")
            t = kpis[(kpis.week_start == pd.Timestamp(wk)) & (kpis.segment != "ALL")]
            st.dataframe(pd.DataFrame({
                "Type": t.segment, "Sales": t.sales.map(eur),
                "vs forecast": t.sales_vs_forecast.map(pct),
                "YoY same-store": t.yoy_growth.map(pct),
                "Basket": t.basket.map(lambda v: f"€{v:.2f}"),
                "Anomaly rate": t.anomaly_rate.map(lambda v: pct(v, sign=False)),
            }), hide_index=True, width="stretch")
        with st.expander("KPI definitions"):
            import yaml
            defs = yaml.safe_load((config.ROOT / "sql" / "kpis.yaml").read_text())
            st.dataframe(pd.DataFrame([{"KPI": d["label"], "Definition": d["definition"],
                                        "Caveat": d.get("caveats", "")} for d in defs]),
                         hide_index=True, width="stretch")


# ================================================================ 2. Cards
def render_card(c: dict) -> None:
    k, q = c["card"], c["qa"]
    with st.container(border=True):
        top = st.columns([6, 2])
        top[0].markdown(f"**{_STRIP.sub('', k.get('title') or c['subject'])}**  \n"
                        f"<span style='color:gray;font-size:0.85em'>{c['subject']}</span>",
                        unsafe_allow_html=True)
        badge = "✅ Published" if q["published"] else "⚠️ Held for review"
        top[1].markdown(f"<div style='text-align:right;font-size:0.85em'>{badge}<br>"
                        f"confidence <b>{k.get('confidence', '?')}</b></div>",
                        unsafe_allow_html=True)
        st.markdown(f"{_STRIP.sub('', k.get('headline', ''))}")
        for b in k.get("bullets", []):
            st.markdown(f"- {_STRIP.sub('', b)}")
        if k.get("action"):
            st.markdown(f"**→ {_STRIP.sub('', k['action'])}**")
        st.caption(f"QA: {q.get('numbers_checked', 0)} numbers checked · accuracy "
                   f"{q.get('numeric_accuracy', 0):.0%} · attempts {q.get('attempts', 1)}"
                   + (f" · issues: {'; '.join(q['open_issues'])}" if q.get("open_issues") else ""))


def live_cards(budget: int) -> tuple[list | None, str | None]:
    from sic.insights import cards as C
    from sic.insights.demo import health_check

    reason = health_check(config.load()["insights"].get("health_check_timeout_s", 10))
    if reason:
        return None, reason
    res: dict = {}

    def work():
        try:
            res["cards"] = C.run()
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {str(e)[:100]}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(budget)
    if t.is_alive():
        return None, f"live generation exceeded {budget}s"
    if "error" in res:
        return None, res["error"]
    rate = sum(c["qa"]["published"] for c in res["cards"]) / max(len(res["cards"]), 1)
    if rate < config.load()["insights"].get("snapshot_min_publish_rate", 0.8):
        return None, f"live run degraded ({rate:.0%} passed QA)"
    return res["cards"], None


with tabs[1]:
    from sic.insights.demo import load_snapshot

    snap = load_snapshot()
    st.write("Weekly briefing cards written by the LLM **only from verified facts**; every "
             "number is fact-checked and a critic reviews claims before a card is published.")
    c1, c2, _ = st.columns([2, 2, 3])
    if c1.button("⚡ Generate live cards", type="primary"):
        budget = config.load()["insights"].get("live_time_budget_s", 240)
        with st.spinner(f"Generating with {config.load()['llm']['provider']} "
                        f"(falls back to saved cards after {budget}s)…"):
            cards, reason = live_cards(budget)
        if cards:
            st.session_state["cards"], st.session_state["cards_src"] = cards, "live"
        else:
            st.session_state["cards_src"] = "snapshot"
            st.warning(f"Live LLM unavailable — {reason}. Showing saved demo cards.")
    if c2.button("Show saved demo cards"):
        st.session_state["cards_src"] = "snapshot"

    src = st.session_state.get("cards_src", "snapshot")
    if src == "live":
        cards = st.session_state["cards"]
        st.success("Live cards generated just now.")
    elif snap:
        cards, m = snap["cards"], snap["meta"]
        st.info(f"Saved demo cards — generated {m['generated_at'][:16].replace('T', ' ')} UTC "
                f"with **{m['model']}** ({m['provider']}), {m['publish_rate']:.0%} passed QA.")
    else:
        cards = None
        st.warning("No saved demo cards yet — click **Generate live cards** while the LLM "
                   "is healthy (a good run is saved automatically for demos).")
    if cards:
        shown = [c for c in cards if c["qa"]["published"]]
        held = len(cards) - len(shown)
        if held:
            st.caption(f"{held} card(s) did not pass QA and are held back for review.")
        cols = st.columns(2)
        for i, c in enumerate(shown):
            with cols[i % 2]:
                render_card(c)


# ================================================================ 3. Stores
with tabs[3]:
    sw, ep, lift, recs = (load("store_weeks.parquet"), load("episode_rca.parquet"),
                          load("promo_lift_by_store.parquet"), load("recommendations.parquet"))
    if sw is None:
        st.warning("Run `make all` first.")
    else:
        flagged = recs[recs.store_id.notna()].store_id.astype(int).tolist() if recs is not None \
            else []
        c1, c2 = st.columns([1, 3])
        sid = c1.selectbox("Store", sorted(sw.store_id.unique()),
                           index=sorted(sw.store_id.unique()).index(flagged[0]) if flagged else 0,
                           help="Defaults to the top-ranked store recommendation.")
        if flagged:
            c2.caption("Stores with an open recommendation: " + ", ".join(map(str, flagged[:15])))
        s = sw[sw.store_id == sid].sort_values("week_start")
        fig = go.Figure()
        fig.add_scatter(x=s.week_start, y=s.forecast, name="Forecast (out-of-sample)",
                        line={"color": NEUTRAL, "width": 2})
        fig.add_scatter(x=s.week_start, y=s.sales, name="Actual", line={"color": BLUE, "width": 2.5})
        a = s[s.is_anomaly]
        fig.add_scatter(x=a.week_start, y=a.sales, mode="markers", name="Anomalous week",
                        marker={"color": ORANGE, "size": 10, "line": {"color": "white", "width": 2}})
        fig.update_yaxes(tickprefix="€")
        plot(st, style(fig, 340).update_layout(title=f"Store {sid}: weekly sales vs forecast"),
                        width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Anomaly episodes & root cause")
            e = ep[ep.store_id == sid].sort_values("start", ascending=False)
            if e.empty:
                st.write("No anomaly episodes in the forecast period.")
            for r in e.head(5).itertuples():
                ev = json.loads(r.evidence)
                with st.container(border=True):
                    st.markdown(f"**{r.start:%d %b %Y} → {r.end:%d %b %Y}** · {pct(r.gap_pct, 0)} "
                                f"({eur(r.gap)}) · {r.weeks} wk · {r.tb_driver}-driven")
                    st.caption(f"Root cause: **{r.root_cause.replace('_', ' ')}**"
                               + "".join(f"  \n• {x['detail']}" for x in ev))
        with c2:
            st.subheader("Promo response & actions")
            if lift is not None and sid in set(lift.store_id):
                L = lift[lift.store_id == sid].iloc[0]
                st.metric("Causal promo lift", pct(L.lift, 0),
                          f"95% CI {pct(L.lift_lo, 0)} to {pct(L.lift_hi, 0)}", delta_color="off")
                st.caption(f"Chain median {pct(lift.lift.median(), 0)}")
            r = recs[recs.store_id == sid] if recs is not None else pd.DataFrame()
            for x in r.itertuples():
                st.success(f"{x.action}  \n*{x.confidence} confidence*")
            if r.empty:
                st.write("No open recommendation for this store.")
            from sic.playbooks import library as PL
            pb = PL.for_store(int(sid))
            for b in pb["playbooks"][:1]:
                with st.container(border=True):
                    st.markdown(f"📘 **Approved playbook: {b['citation']}** — {b['title']}")
                    st.caption(f"Owner {b['owner']} · approved by {b['approved_by']} · "
                               f"review by {b['review_by']}")
                    if "%" in b["expected_impact"]:
                        st.info(b["expected_impact"])
                    for i, play in enumerate(b["plays"], start=1):
                        st.markdown(f"{i}. {play}")
                    if b["do_not"]:
                        st.markdown("**Do not:** " + " · ".join(b["do_not"]))
                    st.caption("Measure: " + "; ".join(b["measure"]))


# ================================================================ 4. Chat + What-if
TOOL_LABELS = {"run_sql": "SQL query", "explain_store": "Store explainer",
               "lookup_definition": "KPI / methodology search", "get_recommendations":
               "Ranked recommendations", "get_playbook": "Approved playbook",
               "simulate_scenario": "What-if simulation"}


def render_sim(r: dict, key: str) -> None:
    """Shared display for a simulation result (chat 'show your work' and the form)."""
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(r["confidence"], "⚪")
    t, w, pc = r["total_eur"], r["per_week_eur"], r["pct_of_baseline"]
    m = st.columns(4)
    m[0].metric("Expected impact", eur(t["mean"]), f"{pct(pc['mean'])} vs baseline",
                delta_color="normal")
    m[1].metric("90% range", f"{eur(t['p5'])} to {eur(t['p95'])}")
    m[2].metric("Per week", eur(w["mean"]))
    m[3].metric("Confidence", f"{conf_icon} {r['confidence']}")
    st.caption(f"{r['description']}: {r['selection']}, {r['weeks']} week(s). Effect used: "
               f"{r['effect_used']}. Method: {r['method']}. {r['uncertainty']}.")
    fig = go.Figure()
    fig.add_bar(y=[str(x["store_id"]) for x in r["top_stores"]][::-1],
                x=[x["expected_eur"] for x in r["top_stores"]][::-1], orientation="h",
                marker_color=BLUE if t["mean"] >= 0 else ORANGE)
    fig.update_xaxes(tickprefix="€")
    fig.update_yaxes(title="Store")
    plot(st, style(fig, 260).update_layout(title="Largest expected impacts by store",
                                                  hovermode="closest"),
                    width="stretch", key=f"simfig_{key}")
    c1, c2 = st.columns(2)
    c1.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in r["assumptions"]))
    c2.markdown("**Caveats**\n" + "\n".join(f"- {c}" for c in r["caveats"]))


def show_your_work(meta: dict, key: str) -> None:
    with st.expander("🔍 Show your work"):
        if not meta.get("tool_calls"):
            st.caption("No tools were called: the answer came from the model alone.")
        for i, c in enumerate(meta.get("tool_calls", []), start=1):
            label = TOOL_LABELS.get(c["tool"], c["tool"])
            st.markdown(f"**{i}. {label}** `{c['tool']}` · {'✅ ok' if c['ok'] else '❌ error'} "
                        f"· {c['latency_s']:.2f}s")
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"raw": c["arguments"]}
            args = {k: v for k, v in args.items() if v is not None}
            if c["tool"] == "run_sql" and "query" in args:
                st.code(args["query"], language="sql")
            else:
                st.code(json.dumps(args, indent=1), language="json")
            try:
                res = json.loads(c.get("result_preview", ""))
            except json.JSONDecodeError:
                res = None
            if isinstance(res, dict) and "rows" in res and "columns" in res:
                st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]).head(20),
                             hide_index=True, width="stretch")
            elif isinstance(res, dict) and c["tool"] == "simulate_scenario" and "total_eur" in res:
                render_sim(res, f"{key}_{i}")
            elif isinstance(res, dict) and res.get("playbooks"):
                for b in res["playbooks"][:2]:
                    st.markdown(f"📘 {b['citation']}: {b['title']}")
            elif res is not None:
                st.json(res, expanded=False)
            elif c.get("result_preview"):
                st.caption(c["result_preview"][:600])
        st.markdown("**Fact-check of the answer**")
        sup, uns = meta.get("supported_numbers", []), meta.get("unsupported_numbers", [])
        st.caption(f"Verified against tool outputs: {', '.join(sup) if sup else '—'}")
        if uns:
            st.warning(f"Not found in any tool output: {', '.join(uns)}")


with tabs[4]:
    chat_tab, sim_tab = st.tabs(["💬 Chat", "🧪 What-if simulator"])

    with chat_tab:
        st.write("Ask about stores, KPIs, programs or **what-if scenarios**. The agent answers "
                 "from tools, and every number is fact-checked — open **Show your work** under "
                 "an answer to see the queries, simulations and sources behind it.")
        examples = ["Why is store 921 below forecast?",
                    "What if a competitor opens 300 m from store 262?",
                    "What if we add a promo week in the 20 lowest-response stores?",
                    "Is the Promo2 effect proven?"]
        ex = st.columns(len(examples))
        for i, q in enumerate(examples):
            if ex[i].button(q, width="stretch", key=f"ex_{i}"):
                st.session_state["pending_q"] = q
        st.session_state.setdefault("chat", [])
        for j, msg in enumerate(st.session_state["chat"]):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("meta"):
                    st.caption(msg["meta"])
                if msg.get("work"):
                    show_your_work(msg["work"], f"hist{j}")
        q = st.chat_input("Ask the Copilot…") or st.session_state.pop("pending_q", None)
        if q:
            st.session_state["chat"].append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            work = None
            with st.chat_message("assistant"), st.spinner("Thinking…"):
                from sic.agent.chat import ask
                hist = [{"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chat"][:-1]][-6:]
                try:
                    t0 = time.time()
                    r = ask(q, history=hist)
                    meta = (f"{r['steps']} tool call(s) · {time.time() - t0:.1f}s · numeric "
                            f"accuracy {r['numeric_accuracy']:.0%}"
                            + (f" · unverified: {', '.join(r['unsupported_numbers'])}"
                               if r["unsupported_numbers"] else ""))
                    answer = r["answer"]
                    work = {k: r.get(k) for k in ("tool_calls", "supported_numbers",
                                                   "unsupported_numbers")}
                except Exception as e:  # noqa: BLE001
                    answer = (f"⚠️ The live LLM is unavailable right now ({type(e).__name__}). "
                              "The dashboards, store explorer, what-if simulator and saved "
                              "insight cards still work.")
                    meta = None
                st.markdown(answer)
                if meta:
                    st.caption(meta)
                if work:
                    show_your_work(work, f"new{len(st.session_state['chat'])}")
            st.session_state["chat"].append({"role": "assistant", "content": answer,
                                             "meta": meta, "work": work})

    with sim_tab:
        from sic.whatif.engine import SCENARIOS, ScenarioError
        from sic.whatif.engine import run as run_sim

        st.write("Run a scenario directly — **no LLM, zero tokens**. Numbers come from causal "
                 "effects measured in the pipeline, with a 90% range from Monte Carlo draws.")
        c1, c2, c3 = st.columns([2, 2, 1])
        scen = c1.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k])
        how = c2.selectbox("Stores", ["Specific stores", "Store type", "Lowest promo response",
                                      "Furthest below forecast", "All stores"])
        weeks = c3.number_input("Weeks", 1, 52, 1 if scen.endswith("promo_week") else 26)
        kw: dict = {"scenario": scen, "weeks": int(weeks)}
        c1, c2 = st.columns(2)
        if how == "Specific stores":
            ids = c1.text_input("Store IDs (comma-separated)", "262, 770")
            kw["store_ids"] = [int(x) for x in re.findall(r"\d+", ids)]
        elif how == "Store type":
            kw["store_type"] = c1.selectbox("Type", ["a", "b", "c", "d"])
        elif how == "Lowest promo response":
            kw["lowest_lift_n"] = int(c1.number_input("How many stores", 1, 1115, 20))
        elif how == "Furthest below forecast":
            kw["weakest_n"] = int(c1.number_input("How many stores", 1, 1115, 10))
        else:
            kw["all_stores"] = True
        if scen == "competitor_opens":
            kw["competitor_distance_m"] = float(c2.number_input(
                "Competitor distance (m)", 10, 20000, 300, step=50))
        if st.button("Run simulation", type="primary"):
            try:
                st.session_state["sim"] = run_sim(**kw)
            except ScenarioError as e:
                st.session_state["sim"] = None
                st.error(str(e))
        if st.session_state.get("sim"):
            render_sim(st.session_state["sim"], "form")


# ================================================================ 5. Model
with tabs[5]:
    bt, bt1 = load_json("m2_backtest_metrics.json"), load_json("m2_backtest_metrics_v1.json")
    bt2 = load_json("m2_backtest_metrics_v2_allfeatures.json")
    pl, did = load_json("m3_promo_lift.json"), load_json("m3_promo2_did.json")
    if bt:
        st.subheader("Forecast accuracy (rolling-origin backtest)")
        def g(m, *keys):
            for k in keys:
                m = m[k] if m else None
            return m
        rows = [("Daily WAPE", ("overall_daily", "baseline", "wape"),
                 ("overall_daily", "p50", "wape")),
                ("Store-week WAPE", ("weekly_store_wape", "baseline"),
                 ("weekly_store_wape", "p50")),
                ("Interval coverage (target 90%)", None, ("overall_daily", "interval_coverage"))]
        cur_n = bt.get("n_features", len(bt.get("features", [])) or 33)
        cols = [("Model v1", bt1), ("v2: all 33 features", bt2 if bt2 is not bt else None),
                (f"Current: {cur_n} selected features" if bt2 else "Current model", bt)]
        st.dataframe(pd.DataFrame([{"Metric": n,
                                    "Baseline": pct(g(bt, *bk), sign=False) if bk else "—",
                                    **{c: pct(g(m, *mk), sign=False) if m else "—"
                                       for c, m in cols}}
                                   for n, bk, mk in rows]),
                     hide_index=True, width="stretch")
        f = pd.DataFrame(bt["folds"])
        fig = go.Figure()
        fig.add_bar(x=f.start, y=[d["wape"] for d in f.baseline], name="Baseline",
                    marker_color=NEUTRAL)
        if bt1:
            fig.add_bar(x=f.start, y=[d["p50"]["wape"] for d in bt1["folds"]], name="Model v1",
                        marker_color=ORANGE)
        if bt2:
            fig.add_bar(x=f.start, y=[d["p50"]["wape"] for d in bt2["folds"]],
                        name="v2: all features", marker_color="#9db8d9")
        fig.add_bar(x=f.start, y=[d["wape"] for d in f.p50], name="Current model",
                    marker_color=BLUE)
        fig.update_yaxes(tickformat=".0%")
        plot(st, style(fig).update_layout(title="Daily WAPE by fold", barmode="group"),
                        width="stretch")
    fs = load_json("m2_feature_selection.json")
    if fs:
        st.subheader("Feature selection")
        st.caption(
            f"Chosen on backtest folds 1-3 ({', '.join(fs['selection_windows'])}); fold "
            f"{fs['settings']['holdout_fold']} (May-Jul 2015) is never seen during selection "
            "and is the held-out comparison. Methods: redundancy clustering (Spearman), out-of-sample permutation "
            "importance (correlated features shuffled together), group ablation, and backward "
            "elimination with retraining. Source: src/sic/forecast/select.py")
        m = st.columns(4)
        ho = load_json("m2_feature_selection_holdout.json")
        m[0].metric("Proposed feature set",
                    f"{len(fs['selected_features'])} of {fs['n_features_full']}",
                    f"{len(fs['dropped_features'])} proposed drops", delta_color="off")
        if ho:
            m[1].metric("Decision on held-out fold", "Adopted" if ho["adopt"] else "Rejected",
                        f"in production: {ho['n_features']['full'] if not ho['adopt'] else ho['n_features']['selected']} features",
                        delta_color="off")
        if ho:
            d = ho["selected"]["store_week_wape"] - ho["full"]["store_week_wape"]
            m[2].metric("Held-out store-week WAPE", pct(ho["selected"]["store_week_wape"], 2,
                                                         sign=False),
                        f"{d * 100:+.2f} pp vs all features", delta_color="inverse")
            dd = ho["selected"]["daily_wape"] - ho["full"]["daily_wape"]
            m[3].metric("Held-out daily WAPE", pct(ho["selected"]["daily_wape"], 2, sign=False),
                        f"{dd * 100:+.2f} pp vs all features", delta_color="inverse")
            lo, hi = ho["delta_daily_wape_ci95"]
            st.info(f"**Held-out fold {ho['holdout_fold']}** (never used to choose features): "
                    f"the {ho['n_features']['selected']}-feature set changes daily WAPE by "
                    f"{dd * 100:+.2f} pp (95% CI {lo * 100:+.2f} to {hi * 100:+.2f}) and "
                    f"store-week WAPE by {d * 100:+.2f} pp. Rule fixed before the test: adopt "
                    "only if not significantly worse. "
                    + ("Adopted." if ho["adopt"] else "Not adopted: the production model keeps "
                       "all features (dropping the proposed ones slightly hurt the held-out fold)."))
        c1, c2 = st.columns(2)
        ga = pd.DataFrame([{"group": g, **v} for g, v in fs["group_ablation"].items()])
        ga = ga.sort_values("delta_wape")
        has_ci = ga.ci95.notna().all()
        fig = go.Figure(go.Bar(
            x=ga.delta_wape * 100, y=ga.group, orientation="h", marker_color=BLUE,
            error_x={"type": "data", "symmetric": False,
                     "array": [(c[1] - v) * 100 for c, v in zip(ga.ci95, ga.delta_wape)],
                     "arrayminus": [(v - c[0]) * 100 for c, v in zip(ga.ci95, ga.delta_wape)],
                     "color": NEUTRAL} if has_ci else None,
            hovertemplate="%{y}: %{x:+.2f} pp WAPE<extra></extra>"))
        fig.update_xaxes(title="WAPE increase when the group is removed (pp"
                         + (", 95% CI)" if has_ci else ")"))
        plot(c1, style(fig, height=380).update_layout(
            title="Group ablation: what each block is worth", hovermode="closest"),
            width="stretch")
        if fs["permutation_importance"]:
            pi = pd.DataFrame([{"unit": u, **v} for u, v in fs["permutation_importance"].items()
                               if len(v["features"]) == 1]).sort_values("mean").tail(15)
            fig = go.Figure(go.Bar(
                x=pi["mean"] * 100, y=pi.unit, orientation="h", marker_color=BLUE,
                error_x={"type": "data", "array": pi["std"] * 100, "color": NEUTRAL},
                hovertemplate="%{y}: %{x:+.2f} pp WAPE<extra></extra>"))
            fig.update_xaxes(title="WAPE increase when shuffled (pp)")
            plot(c2, style(fig, height=380).update_layout(
                title="Permutation importance (top 15, out-of-sample)", hovermode="closest"),
                width="stretch")
        else:
            el = pd.DataFrame(fs["elimination"]).sort_values("delta_wape")
            fig = go.Figure(go.Bar(
                x=el.delta_wape * 100, y=el.feature, orientation="h",
                marker_color=[BLUE if not d else NEUTRAL for d in el.dropped],
                hovertemplate="%{y}: %{x:+.3f} pp WAPE when removed<extra></extra>"))
            fig.update_xaxes(title="WAPE change when removed, retrained (pp); grey = dropped")
            plot(c2, style(fig, height=380).update_layout(
                title="Elimination: cost of removing each candidate", hovermode="closest"),
                width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Backward elimination** (weakest first, retrain after each step)")
            st.dataframe(pd.DataFrame([{
                "Feature": e["feature"], "Decision": "dropped" if e["dropped"] else "kept",
                "Δ WAPE": f"{e['delta_wape'] * 100:+.3f} pp",
                "By fold (pp)": " / ".join(f"{x * 100:+.2f}" for x in e["delta_by_window"])}
                for e in fs["elimination"]]), hide_index=True, width="stretch", height=300)
        with c2:
            st.markdown("**Redundant clusters** (|Spearman ρ| ≥ "
                        f"{fs['redundancy']['threshold']})")
            for cl in fs["redundancy"]["clusters"]:
                kept = [f for f in cl if f in fs["selected_features"]]
                st.write(f"- {', '.join(cl)} → kept: {', '.join(kept) or 'none'}")
            if fs["redundancy"]["near_constant"]:
                st.write("Near-constant on open days: "
                         + ", ".join(fs["redundancy"]["near_constant"]))
            st.write("Dropped: " + (", ".join(fs["dropped_features"]) or "none"))
            if fs.get("partial"):
                st.caption("⚠️ " + fs["note"])

    c1, c2 = st.columns(2)
    if pl:
        lift = load("promo_lift_by_store.parquet")
        fig = go.Figure(go.Histogram(x=lift.lift, nbinsx=40, marker_color=BLUE))
        fig.update_xaxes(tickformat=".0%")
        plot(c1, style(fig).update_layout(
            title=f"Causal promo lift per store (pooled {pct(pl['pooled_lift'])})",
            hovermode="closest"), width="stretch")
    if did:
        es = pd.DataFrame(did["event_study"])
        fig = go.Figure()
        for mask, color, name in ((es.bin < 0, NEUTRAL, "Before enrollment"),
                                  (es.bin >= 0, BLUE, "After enrollment")):
            d = es[mask]
            fig.add_scatter(x=d.bin * 4, y=d.coef, mode="markers", name=name,
                            marker={"color": color, "size": 9},
                            error_y={"type": "data", "symmetric": False, "array": d.hi - d.coef,
                                         "arrayminus": d.coef - d.lo, "color": color})
        fig.update_xaxes(title="Weeks relative to Promo2 enrollment")
        plot(c2, style(fig, pct_axis=True).update_layout(
            title=f"Promo2 event study: {pct(did['effect']['effect_pct'])} "
                  f"(pre-trend p={did['pretrend_test']['p_value']:.3f}: suggestive)",
            hovermode="closest"), width="stretch")
    st.caption("Full write-up: reports/REVIEW_M1_M3.md")


# ================================================================ 6. Evals
with tabs[6]:
    ret, qa, qal = (load_json("m5_retrieval_eval.json"), load_json("m5_card_qa_eval.json"),
                    load_json("m5_card_qa_eval_llm.json"))
    ag, ab, dr, ob = (load_json("m5_agent_eval.json"), load_json("m5_ab_design.json"),
                      load_json("m5_drift.json"), load_json("m5_observability.json"))
    st.write("How we know the system works — refreshed by `make eval` (offline) and "
             "`make eval-live` (LLM). Full write-up: reports/M5_EVALS.md")

    m = st.columns(4)
    if ag:
        m[0].metric("Agent answers correct", pct(ag["correctness"], 0, sign=False),
                    f"{ag['n']} golden questions", delta_color="off")
        m[1].metric("Agent latency p50 / p95", f"{ag['latency_p50_s']:.0f}s / "
                    f"{ag['latency_p95_s']:.0f}s")
    if qa:
        full = qal["full_gate_recall"] if qal else None
        m[2].metric("Card QA: defects caught", pct(full if full is not None
                                                    else qa["deterministic_recall"], 0, sign=False),
                    "rules + LLM critic" if full is not None else "rules only", delta_color="off")
        m[3].metric("Card QA: false alarms", pct(qa["deterministic_false_positive_rate"], 0,
                                                 sign=False), "on correct cards", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Card QA gate (injected defects)")
        if qa:
            llm_by = qal["by_defect"] if qal else {}
            st.dataframe(pd.DataFrame([{
                "Defect": "clean (false alarms)" if d == "clean" else d.replace("_", " "),
                "Cards": v["n"], "Rules": pct(v["deterministic"], 0, sign=False),
                "LLM critic": pct(llm_by.get(d, {}).get("llm_critic"), 0, sign=False),
                "Full gate": pct(llm_by.get(d, {}).get("full_gate"), 0, sign=False)}
                for d, v in qa["by_defect"].items()]), hide_index=True, width="stretch")
    with c2:
        st.subheader("Agent golden set")
        if ag:
            st.dataframe(pd.DataFrame([{
                "": "✅" if r["correct"] else "❌", "Question": r["q"],
                "Tools": ", ".join(r["tools_used"]), "Numeric acc.":
                pct(r["numeric_accuracy"], 0, sign=False), "Latency": f"{r['latency_s']:.0f}s"}
                for r in ag["rows"]]), hide_index=True, width="stretch")
        if ret:
            st.caption("Retrieval (held-out paraphrases): dense R@1 "
                       f"{pct(ret['dense']['holdout']['recall@1'], 0, sign=False)} vs lexical "
                       f"{pct(ret['lexical']['holdout']['recall@1'], 0, sign=False)}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("A/B design: promo pilot")
        if ab:
            from scipy.stats import norm
            fig = go.Figure()
            eff = [i / 1000 for i in range(151)]
            for name, color, label in (("raw", NEUTRAL, "Raw sales"), ("vs_fc", BLUE,
                                       "Sales vs forecast"), ("cuped", AQUA, "+ CUPED")):
                sd = ab["by_duration_promo_cycles"]["2"]["placebo_sd"][name]
                fig.add_scatter(x=eff, y=[norm.cdf(math.log1p(e) / sd - 1.96) for e in eff],
                                name=label, line={"color": color, "width": 2.5})
            fig.add_hline(y=0.8, line_dash="dot", line_color=NEUTRAL)
            fig.update_xaxes(tickformat=".0%", title="True effect (2 promo cycles)")
            fig.update_yaxes(tickformat=".0%", title="Power")
            plot(st, style(fig).update_layout(hovermode="closest"), width="stretch")
            st.caption(ab["recommendation"])
    with c2:
        st.subheader("Drift & model health")
        if dr:
            alerts = dr["performance"]["control_limit_alerts"]
            for a in alerts:
                if "rule" in a:
                    st.warning(f"**{a['weeks']} weeks of {a['direction']}** through "
                               f"{a['through']}: retraining trigger.")
            wk = pd.DataFrame(dr["weekly"])
            fig = go.Figure()
            fig.add_scatter(x=wk.week_start, y=wk.wape, name="WAPE", line={"color": BLUE})
            fig.add_scatter(x=wk.week_start, y=wk.bias, name="Bias (+ = over-forecast)",
                            line={"color": ORANGE})
            fig.update_yaxes(tickformat=".0%")
            plot(st, style(fig, 260), width="stretch")
            st.dataframe(pd.DataFrame(dr["features"]).rename(columns={
                "psi_vs_training_year": "PSI vs training", "psi_vs_same_weeks_last_year":
                "PSI vs last year", "status_seasonal_adjusted": "status (seasonal)"}),
                hide_index=True, width="stretch")
    if ob:
        st.subheader("LLM providers tried (from call logs)")
        st.dataframe(pd.DataFrame(ob["by_provider_model"])[[
            "provider", "model", "calls", "error_rate", "rate_limited", "truncated",
            "latency_p50_s", "latency_p95_s"]].round(3), hide_index=True, width="stretch")


# ================================================================ Opportunities
with tabs[2]:
    opp, oj = load("opportunity_scores.parquet"), load_json("m3_opportunity.json")
    if opp is None:
        st.warning("Run `uv run python -m sic.opportunity.score` first.")
    else:
        st.write("Which stores have the most upside from management attention? Each store's "
                 "score is a **risk-adjusted weekly € upside** from three levers. Causal levers "
                 "count fully; the observational one is down-weighted.")
        m = st.columns(4)
        m[0].metric("Top-10 upside / week", eur(opp.head(10).score_eur.sum()))
        m[1].metric("Grow with promos", f"{(opp.primary_lever == 'promo').sum()} stores")
        m[2].metric("Lift basket to peer level", f"{(opp.primary_lever == 'peer').sum()} stores")
        m[3].metric("Defend vs competitor", f"{(opp.primary_lever == 'competitor').sum()} stores")

        f1, f2, f3, f4 = st.columns(4)
        lever = f1.selectbox("Main lever", ["All", "promo", "peer", "competitor"],
                             format_func=lambda x: {"All": "All", "promo": "Grow with promos",
                                                    "peer": "Lift basket to peer level",
                                                    "competitor": "Defend vs competitor"}[x])
        stype = f2.selectbox("Store type", ["All", "a", "b", "c", "d"])
        mom = f3.selectbox("Momentum", ["All", "sliding", "stable", "improving"])
        n = f4.slider("Show top", 5, 50, 15)
        v = opp.copy()
        if lever != "All":
            v = v[v.primary_lever == lever]
        if stype != "All":
            v = v[v.store_type == stype]
        if mom != "All":
            v = v[v.momentum == mom]
        v = v.head(n)
        w = config.load()["opportunity"]["confidence_weights"]

        fig = go.Figure()
        ylab = [f"#{r} · store {s_}" for r, s_ in zip(v["rank"], v.store_id, strict=False)][::-1]
        for col, color, name, wt in (("eur_promo", BLUE, "Promo lever", w["promo"]),
                                     ("eur_peer", ORANGE, "Basket vs peers", w["peer"]),
                                     ("eur_competitor", AQUA, "Competitor defense",
                                      w["competitor"])):
            fig.add_bar(y=ylab, x=(v[col] * wt)[::-1], name=name, orientation="h",
                        marker_color=color, marker_line_color="white", marker_line_width=1)
        fig.update_xaxes(tickprefix="€", title="Risk-adjusted weekly upside")
        plot(st, style(fig, max(260, 26 * len(v) + 80)).update_layout(
            title="Opportunity score by lever", barmode="stack", hovermode="y unified"),
            width="stretch")

        st.dataframe(pd.DataFrame({
            "Rank": v["rank"], "Store": v.store_id, "Type": v.store_type,
            "Upside / week": v.score_eur.map(eur), "Main lever": v.opportunity_type,
            "Momentum": v.momentum, "Why": v.why}), hide_index=True, width="stretch")

        st.subheader("Store detail")
        c1, c2 = st.columns([1, 3])
        sid = c1.selectbox("Store", v.store_id.tolist() or opp.store_id.head(1).tolist(),
                           key="opp_store")
        r = opp[opp.store_id == sid].iloc[0]
        c2.markdown(f"**#{r['rank']} · store {sid} · {r.opportunity_type}**  \n{r.why}")
        d1, d2 = st.columns(2)
        with d1:
            peers = opp[opp.store_id.isin(list(r.peers))][["store_id", "basket", "spd"]]
            if len(peers):
                st.caption(f"Look-alike stores (same type {r.store_type}, assortment; similar "
                           f"competition and Promo2): median basket €{r.peer_basket:.2f} vs "
                           f"this store €{r.basket:.2f}")
                st.dataframe(peers.rename(columns={"store_id": "Peer store", "basket": "Basket €",
                                                   "spd": "Sales / open day €"}).round(2),
                             hide_index=True, width="stretch")
        with d2:
            st.caption("Size an action with the what-if engine (no LLM, zero tokens)")
            if st.button(f"Simulate one extra promo week in store {sid}"):
                from sic.whatif.engine import run as run_sim
                st.session_state["opp_sim"] = run_sim(scenario="extra_promo_week",
                                                      store_ids=[int(sid)])
        if st.session_state.get("opp_sim") and st.session_state["opp_sim"]["inputs"].get(
                "store_ids") == [int(sid)]:
            render_sim(st.session_state["opp_sim"], "opp")

        if oj and oj.get("validation"):
            val = oj["validation"]
            with st.expander("How we know the levers are real (validation & assumptions)"):
                st.markdown(
                    f"- **Promo lift is a stable store trait**: per-store lift estimated on two "
                    f"independent periods ranks stores the same way (Spearman "
                    f"**{val['promo_lift_split_sample_spearman']:.2f}**, "
                    f"{val['promo_lift_stores']} stores).\n"
                    f"- **Basket gap vs peers is persistent** (year-over-year Spearman "
                    f"**{val['basket_gap_year_over_year_spearman']:.2f}**) and only weakly tied "
                    f"to store size ({val['basket_gap_vs_store_size_spearman']:.2f}), so it is "
                    f"not just location.\n"
                    f"- **Rejected**: {val['rejected_lever']}.\n"
                    f"- Assumptions (config.yaml): capture rate "
                    f"{oj['assumptions']['peer_capture_rate']:.0%} of the basket gap; one extra "
                    f"promo week per quarter; {oj['assumptions']['competitor_protectable_share']:.0%}"
                    f" of the expected competitor loss is defensible; weights {w}.")
