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


# ---------------------------------------------------------------- header
st.title("Sales Insight Copilot")
st.caption("Detect what changed, diagnose why, recommend what to do — every number traceable. "
           "Rossmann data (1,115 stores, through 2015-07-31).")

kpis = load("weekly_kpis.parquet")
tabs = st.tabs(["📊 Weekly KPIs", "🗂 Insight cards", "🏪 Store explorer", "💬 Ask the Copilot",
                "🔬 Model & methods", "✅ Evals & monitoring"])

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
        g1.plotly_chart(style(fig, pct_axis=True).update_layout(title="Sales vs forecast"),
                        width="stretch")
        ky = k.dropna(subset=["yoy_growth"])
        ky = ky[~ky.partial_week]
        fig = go.Figure()
        fig.add_bar(x=ky.week_start, y=ky.yoy_customers_growth, name="Customers", marker_color=BLUE)
        fig.add_bar(x=ky.week_start, y=ky.yoy_basket_growth, name="Basket", marker_color=ORANGE)
        g2.plotly_chart(style(fig, pct_axis=True).update_layout(
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
    c1, c2, c3 = st.columns([2, 2, 3])
    show_held = c3.toggle("Show held cards", value=False)
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
        shown = [c for c in cards if show_held or c["qa"]["published"]]
        cols = st.columns(2)
        for i, c in enumerate(shown):
            with cols[i % 2]:
                render_card(c)


# ================================================================ 3. Stores
with tabs[2]:
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
        st.plotly_chart(style(fig, 340).update_layout(title=f"Store {sid}: weekly sales vs forecast"),
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


# ================================================================ 4. Chat
with tabs[3]:
    st.write("Ask about stores, KPIs or programs. The agent answers from tools (SQL, store "
             "explainer, KPI docs) and every number in its answer is fact-checked.")
    examples = ["Why is store 921 below forecast?", "What does traffic-driven mean?",
                "Which store types had the highest same-store YoY growth last week?",
                "Is the Promo2 effect proven?"]
    ex = st.columns(len(examples))
    for i, q in enumerate(examples):
        if ex[i].button(q, width="stretch"):
            st.session_state["pending_q"] = q
    st.session_state.setdefault("chat", [])
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
    q = st.chat_input("Ask the Copilot…") or st.session_state.pop("pending_q", None)
    if q:
        st.session_state["chat"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            from sic.agent.chat import ask
            hist = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state["chat"][:-1]][-6:]
            try:
                t0 = time.time()
                r = ask(q, history=hist)
                meta = (f"{r['steps']} tool call(s) · {time.time() - t0:.1f}s · numeric accuracy "
                        f"{r['numeric_accuracy']:.0%}"
                        + (f" · unverified: {', '.join(r['unsupported_numbers'])}"
                           if r["unsupported_numbers"] else ""))
                answer = r["answer"]
            except Exception as e:  # noqa: BLE001
                answer = (f"⚠️ The live LLM is unavailable right now ({type(e).__name__}). "
                          "The dashboards, store explorer and saved insight cards still work.")
                meta = None
            st.markdown(answer)
            if meta:
                st.caption(meta)
        st.session_state["chat"].append({"role": "assistant", "content": answer, "meta": meta})


# ================================================================ 5. Model
with tabs[4]:
    bt, bt1 = load_json("m2_backtest_metrics.json"), load_json("m2_backtest_metrics_v1.json")
    pl, did = load_json("m3_promo_lift.json"), load_json("m3_promo2_did.json")
    if bt:
        st.subheader("Forecast accuracy (rolling-origin backtest)")
        rows = [("Daily WAPE", bt["overall_daily"]["baseline"]["wape"],
                 bt1["overall_daily"]["p50"]["wape"] if bt1 else None,
                 bt["overall_daily"]["p50"]["wape"]),
                ("Store-week WAPE", bt["weekly_store_wape"]["baseline"],
                 bt1["weekly_store_wape"]["p50"] if bt1 else None, bt["weekly_store_wape"]["p50"]),
                ("Interval coverage (target 90%)", None,
                 bt1["overall_daily"]["interval_coverage"] if bt1 else None,
                 bt["overall_daily"]["interval_coverage"])]
        st.dataframe(pd.DataFrame([{"Metric": n, "Baseline": pct(b, sign=False) if b else "—",
                                    "Model v1": pct(v1, sign=False) if v1 else "—",
                                    "Current model": pct(v2, sign=False)}
                                   for n, b, v1, v2 in rows]),
                     hide_index=True, width="stretch")
        f = pd.DataFrame(bt["folds"])
        fig = go.Figure()
        fig.add_bar(x=f.start, y=[d["wape"] for d in f.baseline], name="Baseline",
                    marker_color=NEUTRAL)
        if bt1:
            fig.add_bar(x=f.start, y=[d["p50"]["wape"] for d in bt1["folds"]], name="Model v1",
                        marker_color=ORANGE)
        fig.add_bar(x=f.start, y=[d["wape"] for d in f.p50], name="Current model",
                    marker_color=BLUE)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig).update_layout(title="Daily WAPE by fold", barmode="group"),
                        width="stretch")
    c1, c2 = st.columns(2)
    if pl:
        lift = load("promo_lift_by_store.parquet")
        fig = go.Figure(go.Histogram(x=lift.lift, nbinsx=40, marker_color=BLUE))
        fig.update_xaxes(tickformat=".0%")
        c1.plotly_chart(style(fig).update_layout(
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
        c2.plotly_chart(style(fig, pct_axis=True).update_layout(
            title=f"Promo2 event study: {pct(did['effect']['effect_pct'])} "
                  f"(pre-trend p={did['pretrend_test']['p_value']:.3f}: suggestive)",
            hovermode="closest"), width="stretch")
    st.caption("Full write-up: reports/REVIEW_M1_M3.md")


# ================================================================ 6. Evals
with tabs[5]:
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
            st.plotly_chart(style(fig).update_layout(hovermode="closest"), width="stretch")
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
            st.plotly_chart(style(fig, 260), width="stretch")
            st.dataframe(pd.DataFrame(dr["features"]).rename(columns={
                "psi_vs_training_year": "PSI vs training", "psi_vs_same_weeks_last_year":
                "PSI vs last year", "status_seasonal_adjusted": "status (seasonal)"}),
                hide_index=True, width="stretch")
    if ob:
        st.subheader("LLM providers tried (from call logs)")
        st.dataframe(pd.DataFrame(ob["by_provider_model"])[[
            "provider", "model", "calls", "error_rate", "rate_limited", "truncated",
            "latency_p50_s", "latency_p95_s"]].round(3), hide_index=True, width="stretch")
