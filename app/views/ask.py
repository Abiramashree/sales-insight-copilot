"""Ask: chat with the copilot, or run a what-if scenario directly (no LLM)."""

from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st
from ui import simulation

TOOLS = {"run_sql": "SQL", "explain_store": "Store explainer", "lookup_definition": "Definitions",
         "get_recommendations": "Recommendations", "get_playbook": "Playbook",
         "simulate_scenario": "Simulation", "get_opportunities": "Opportunities"}
EXAMPLES = ["Why is store 921 below forecast?", "Which stores should we prioritize?",
            "What if a competitor opens 300 m from store 262?", "Is the Promo2 effect proven?"]

mode = st.segmented_control("Mode", ["Chat", "Simulate"], default="Chat", required=True,
                            label_visibility="collapsed")


# ---------------------------------------------------------------- chat
def sources(work: dict, key: str) -> None:
    """What the answer was built from: each tool call with its input and result."""
    calls = work.get("tool_calls") or []
    if not calls:
        return
    with st.expander(f"Sources · {len(calls)}"):
        for i, c in enumerate(calls, 1):
            st.markdown(f"**{TOOLS.get(c['tool'], c['tool'])}**"
                        + ("" if c["ok"] else " · failed"))
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
            elif isinstance(res, dict) and "total_eur" in res:
                simulation(res, f"{key}_{i}")
            elif res is not None:
                st.json(res, expanded=False)


def unverified(work: dict) -> None:
    if work.get("unsupported_numbers"):
        st.warning("Not found in any source: " + ", ".join(work["unsupported_numbers"]))


if mode == "Chat":
    history = st.session_state.setdefault("chat", [])
    if not history:
        picked = st.pills("Examples", EXAMPLES, label_visibility="collapsed", key="example")
        if picked:
            st.session_state["pending"] = picked
    for j, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("work"):
                unverified(msg["work"])
                sources(msg["work"], f"h{j}")

    q = st.chat_input("Ask about stores, KPIs or scenarios") or st.session_state.pop(
        "pending", None)
    if q:
        history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            work = None
            with st.spinner("Thinking…"):
                from sic.agent.chat import ask

                try:
                    r = ask(q, history=[{"role": m["role"], "content": m["content"]}
                                        for m in history[:-1]][-6:])
                    answer = r["answer"]
                    work = {k: r.get(k) for k in ("tool_calls", "unsupported_numbers")}
                except Exception as e:  # noqa: BLE001 - provider down: say so, keep the app usable
                    answer = (f"The live model is unavailable ({type(e).__name__}). "
                              "Week, Stores and Simulate still work.")
            st.markdown(answer)
            if work:
                unverified(work)
                sources(work, f"h{len(history)}")
        history.append({"role": "assistant", "content": answer, "work": work})

# ---------------------------------------------------------------- simulate
else:
    from sic.whatif.engine import SCENARIOS, ScenarioError
    from sic.whatif.engine import run as run_sim

    SELECT = {"ids": "Specific stores", "type": "Store type", "lift": "Lowest promo response",
              "weak": "Furthest below forecast", "all": "All stores"}
    c1, c2, c3 = st.columns([3, 2, 1])
    scen = c1.selectbox("Scenario", list(SCENARIOS), format_func=SCENARIOS.get)
    how = c2.selectbox("Stores", list(SELECT), format_func=SELECT.get)
    weeks = c3.number_input("Weeks", 1, 52, 1 if scen.endswith("promo_week") else 26)
    kw: dict = {"scenario": scen, "weeks": int(weeks)}
    c1, c2, _ = st.columns([3, 2, 1])
    if how == "ids":
        kw["store_ids"] = [int(x) for x in re.findall(r"\d+", c1.text_input("Store IDs",
                                                                           "262, 770"))]
    elif how == "type":
        kw["store_type"] = c1.selectbox("Type", ["a", "b", "c", "d"])
    elif how == "lift":
        kw["lowest_lift_n"] = int(c1.number_input("Number of stores", 1, 1115, 20))
    elif how == "weak":
        kw["weakest_n"] = int(c1.number_input("Number of stores", 1, 1115, 10))
    else:
        kw["all_stores"] = True
    if scen == "competitor_opens":
        kw["competitor_distance_m"] = float(c2.number_input("Competitor distance (m)", 10,
                                                            20000, 300, step=50))
    if st.button("Run", type="primary"):
        try:
            st.session_state["sim"] = run_sim(**kw)
        except ScenarioError as e:
            st.session_state["sim"] = None
            st.error(str(e))
    if st.session_state.get("sim"):
        simulation(st.session_state["sim"], "form")
