"""Assistant: a junior BA. One chat for questions and plans; every number in an answer
comes from the deterministic engine (planner, tests, A/B design and readout)."""

from __future__ import annotations

import json

import streamlit as st
from ui import plan_block, readout, sources, unverified

from sic import planner

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


# ---------------------------------------------------------------- chat
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
                answer = (f"The live model is unavailable ({type(e).__name__}). "
                          "Try again shortly; the KPIs page still works.")
        st.markdown(answer)
        if work:
            unverified(work)
            rich(work, f"h{len(history)}")
            sources(work, f"h{len(history)}")
    history.append({"role": "assistant", "content": answer, "work": work})
