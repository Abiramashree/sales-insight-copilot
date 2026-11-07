"""Assistant: a junior BA. Chat for questions and plans; a manual panel (no LLM) to test
hypotheses, design an A/B and read out results. Every number comes from the engine."""

from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st
from ui import pct, plan_block, readout, sources, test_result, unverified

from sic import config, planner
from sic import experiment as X
from sic import hypothesis as H

EXAMPLES = ["Grow type b sales 5% next quarter", "Which stores fell most last month?",
            "Test an extra promo week in 20 weak stores"]


@st.cache_data(show_spinner=False)
def _plan(args_json: str) -> dict:
    a = json.loads(args_json)
    return planner.plan(a["target_pct"], store_type=a.get("store_type"),
                        store_ids=a.get("store_ids"), test_weeks=int(a.get("test_weeks") or 8))


def rich(work: dict, key: str) -> None:
    """Render the deterministic result behind the answer (plan or readout), if any."""
    for i, c in enumerate(work.get("tool_calls") or []):
        if not c["ok"]:
            continue
        if c["tool"] == "plan_initiative":
            plan_block(_plan(c["arguments"]), f"{key}_{i}")
        elif c["tool"] == "analyze_ab_test":
            try:
                readout(json.loads(c["result_preview"]))
            except (json.JSONDecodeError, KeyError):
                pass


chat, panel = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------- chat
with chat:
    history = st.session_state.setdefault("chat", [])
    if not history and "pending" not in st.session_state:
        picked = st.pills("Examples", EXAMPLES, label_visibility="collapsed", key="example",
                          wrap=True)
        if picked:
            st.session_state["pending"] = picked
    for j, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("work"):
                unverified(msg["work"])
                rich(msg["work"], f"h{j}")
                sources(msg["work"], f"h{j}")
    q = st.chat_input("Ask a question or describe a goal") or st.session_state.pop("pending", None)
    if q:
        history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            work = None
            with st.spinner("Working…"):
                from sic.agent.chat import ask

                try:
                    r = ask(q, history=[{"role": m["role"], "content": m["content"]}
                                        for m in history[:-1]][-8:])
                    answer = r["answer"]
                    work = {k: r.get(k) for k in ("tool_calls", "unsupported_numbers")}
                except Exception as e:  # noqa: BLE001 - provider down: say so, panel still works
                    answer = (f"The live model is unavailable ({type(e).__name__}). The manual "
                              "panel on the right still runs tests, designs and readouts.")
            st.markdown(answer)
            if work:
                unverified(work)
                rich(work, f"h{len(history)}")
                sources(work, f"h{len(history)}")
        history.append({"role": "assistant", "content": answer, "work": work})

# ---------------------------------------------------------------- manual panel (no LLM)
with panel:
    tool = st.segmented_control("Tool", ["Test", "Design", "Analyze"], default="Test",
                                required=True, label_visibility="collapsed")

    if tool == "Test":
        COMPARE = {**{k: f"{a} vs {b}" for k, (a, b) in H.WITHIN.items()}, **H.BETWEEN}
        metric = st.selectbox("Metric", list(H.METRICS), format_func=H.METRICS.get)
        comp = st.selectbox("Compare", list(COMPARE), format_func=COMPARE.get)
        a = b = stype = None
        if comp in H.BETWEEN:
            opts = H.groups(comp)
            g1, g2 = st.columns(2)
            a = g1.selectbox("Group", opts, key=f"a_{comp}")
            b = g2.selectbox("Against", [o for o in opts if o != a], key=f"b_{comp}_{a}")
        else:
            t = st.selectbox("Stores", ["All", "a", "b", "c", "d"],
                             format_func=lambda v: v if v == "All" else f"Type {v}")
            stype = None if t == "All" else t
        try:
            r = H.test(metric, comp, a, b, store_type=stype)
            st.markdown(f"**{r['a']} vs {r['b']}**")
            test_result(r)
        except ValueError as e:
            st.warning(str(e))

    elif tool == "Design":
        how = st.segmented_control("Stores", ["Top upside", "Store type", "Store IDs"],
                                   default="Top upside", required=True,
                                   label_visibility="collapsed")
        opp = pd.read_parquet(config.path("outputs") / "opportunity_scores.parquet")
        if how == "Top upside":
            n = st.number_input("Number of stores", 4, 1115, 40, step=2)
            ids = opp.sort_values("score_eur", ascending=False).head(int(n)).store_id.tolist()
        elif how == "Store type":
            t = st.selectbox("Type", ["a", "b", "c", "d"])
            ids = opp[opp.store_type == t].store_id.tolist()
        else:
            ids = [int(i) for i in re.findall(r"\d+", st.text_input("Store IDs", "262, 770, 1114, 817"))]
        c1, c2 = st.columns(2)
        lift = c1.number_input("Lift to detect (%)", 0.5, 50.0, 3.0, step=0.5) / 100
        weeks = c2.slider("Weeks", 2, 12, 8)
        try:
            groups = X.assign(ids)
            size = H.plan(weeks=weeks, stores_per_arm=len(ids) // 2, target_lift=lift)
        except ValueError as e:
            st.warning(str(e))
            st.stop()
        ok = size["stores_needed_cuped"] <= len(ids) // 2
        st.markdown(":green-badge[Powered]" if ok else
                    f":orange-badge[Under-powered: needs {size['stores_needed_cuped']} stores "
                    "per group]")
        m = st.columns(2)
        m[0].metric("Detectable lift", pct(size["mde_cuped"]),
                    f"{pct(size['mde_raw'])} without CUPED", delta_color="off", delta_arrow="off")
        m[1].metric("Stores per group", len(ids) // 2,
                    f"{size['stores_needed_cuped']} needed", delta_color="off", delta_arrow="off")
        st.dataframe(groups.rename(columns={"store_id": "Store", "group": "Group",
                                            "weekly_sales": "Weekly sales"}),
                     hide_index=True, width="stretch", height=220)
        st.session_state["design"] = {"ids": ids, "lift": lift, "weeks": weeks}

    else:
        src = st.segmented_control("Results", ["Dry run", "Upload CSV"], default="Dry run",
                                   required=True, label_visibility="collapsed")
        d = st.session_state.get("design")
        if src == "Dry run":
            if not d:
                st.write("Design a test first; the dry run reads it out on past weeks.")
                st.stop()
            w = sorted(X.weekly().week_start.unique())
            starts = [s for s in w[: len(w) - d["weeks"] + 1]][-26:]
            start = st.selectbox("Start week", starts[::-1],
                                 format_func=lambda s: f"{pd.Timestamp(s):%-d %b %Y}")
            try:
                readout(X.dry_run(X.assign(d["ids"]), str(pd.Timestamp(start).date()),
                                  d["weeks"], d["lift"]))
            except ValueError as e:
                st.warning(str(e))
        else:
            f = st.file_uploader("Results CSV: store_id, week_start, group, sales", type="csv")
            if f is not None:
                try:
                    readout(X.analyze(X.read_csv(f.getvalue()), d["lift"] if d else None))
                except ValueError as e:
                    st.warning(str(e))
