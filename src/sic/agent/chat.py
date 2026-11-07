"""M4: tool-using sales-insights chat agent (provider set in config.yaml → llm).

Loop: model -> (tool calls -> results)* -> answer, capped at `max_steps`.
Each run is traced to data/outputs/agent_traces.jsonl with tool calls,
latency, and a numeric fact-check of the answer against tool outputs.

  uv run python -m sic.agent.chat "Why is store 921 below forecast?"
  uv run python -m sic.agent.chat            # interactive
"""

from __future__ import annotations

import json
import sys
import time

from sic import config
from sic.agent import tools as T
from sic.insights.factcheck import check, numbers_in

SYSTEM = """You are the Sales Insight Copilot, an analyst assistant for a retail chain's sales
managers (1,115 stores, daily data through {data_end}, amounts in EUR).

How to work:
- Get every number from a tool. Never guess or do mental arithmetic on large figures; if you
  need a calculation, do it in SQL.
- For "why" questions about a store, call explain_store first. For definitions or "how is X
  measured", call lookup_definition. For ranked actions across stores, call
  get_recommendations. For "what should we do / how do we respond" (a store, a competitor
  opening, a type of drop) and for the typical impact of competitor openings, call
  get_playbook and cite the playbook (e.g. "PB-COMP-NEAR v2"). Never invent actions or
  impacts that no tool returned.
- For "which stores should we prioritize / where is the most upside", call
  get_opportunities and explain the top stores' reasons; offer to size an action with
  simulate_scenario.
- For "what if" / "what would happen" questions (adding or cancelling promos, a competitor
  opening, enrolling stores in Promo2), call simulate_scenario. Report the expected impact,
  the 90% range, the confidence level and the main caveat. If the scenario is not one of the
  supported ones, say so and list what can be simulated.
- To check whether a metric really differs between conditions (promo, school holiday, store
  type, assortment, Promo2, competitor distance), call test_hypothesis; to size an A/B
  test, call plan_ab_test. Say whether a test is within-store or an association only.
- Planning a goal ("grow / increase / reach ..."): you are the BA's junior analyst. If the
  target %, the scope (chain, a store type, or specific stores) or the time frame is
  missing, ask ONE short clarifying question that offers a sensible default, then stop.
  Otherwise call plan_initiative and write the plan under these headings, in order:
  **Goal**, **Is it plausible?**, **Stores and targets**, **Hypotheses and tests**,
  **Experiment**, **Alternatives**, **Risks and decision**. Use only numbers the tools
  return: state the verdict with its coverage, give the number of stores selected
  (stores_selected) even if fewer are listed, quote CIs, say if the experiment is
  under-powered and what fixes it (more stores, more weeks, a larger effect), and note
  that the basket alternative's range is assumption-based. When the BA refines the plan
  ("only type b", "8 weeks"), call plan_initiative again with the new inputs.
- Experiments: design_ab_test sizes a test and splits stores; analyze_ab_test reads out a
  dry run on history (no treatment, so a sound design shows ~0).
- Say how certain a finding is. Flag early signals (1-2 weeks) and estimates the docs call
  uncertain (e.g. the Promo2 effect is suggestive, not proven).
- Do not invent causes. If the root cause is "unexplained", say local investigation is needed.
- If a question is outside the data, say so plainly.

Answer style (questions): lead with the answer in 1-2 sentences, then up to 4 short bullets
of evidence, then a suggested next step if relevant. Plans use the headings above.
Keep numbers exactly as the tools return them (you may format 0.231 as 23.1% and 13326 as
€13.3k).

Tables you can query with run_sql:
{schema}
"""


def data_end() -> str:
    return str(config.as_of() or config.load()["feed"]["end"])


def _system() -> str:
    return SYSTEM.format(schema=T.schema_description(), data_end=data_end())


def _tool_call_dict(c) -> dict:
    """Echo a tool call back to the provider unchanged. Gemini 3 attaches a thought
    signature (extra_content) that must be returned on the next turn, or it answers 400."""
    if hasattr(c, "model_dump"):
        return c.model_dump(exclude_none=True)
    return {"id": c.id, "type": "function",
            "function": {"name": c.function.name, "arguments": c.function.arguments}}


def _call_tool(name: str, args_json: str) -> tuple[dict, bool]:
    try:
        args = json.loads(args_json or "{}")
        if name not in T.TOOLS:
            return {"error": f"Unknown tool {name}"}, False
        return T.TOOLS[name](**args), True
    except T.SQLError as e:
        return {"error": str(e)}, False
    except (TypeError, json.JSONDecodeError) as e:
        return {"error": f"Bad arguments for {name}: {e}"}, False


def ask(question: str, llm=None, history: list[dict] | None = None,
        system: str | None = None, max_steps: int | None = None) -> dict:
    """Run the tool loop. `system` replaces the copilot brief (the test advisor uses this);
    tools, tracing and the numeric fact-check are the same either way."""
    from sic.llm.client import get_llm

    llm = llm or get_llm()
    cfg = config.load()["agent"]
    messages = [{"role": "system", "content": system or _system()}, *(history or []),
                {"role": "user", "content": question}]
    trace, evidence, t0 = [], [], time.time()
    answer = None
    for step in range(max_steps or cfg["max_steps"]):
        msg = llm.chat(messages, tools=T.TOOL_SPECS, purpose="agent")
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            answer = msg.content or ""
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [_tool_call_dict(c) for c in calls]})
        for c in calls:
            ts = time.time()
            result, ok = _call_tool(c.function.name, c.function.arguments)
            payload = json.dumps(result, default=str)
            if len(payload) > cfg["tool_result_max_chars"]:
                payload = payload[: cfg["tool_result_max_chars"]] + '..."(truncated)"'
            messages.append({"role": "tool", "tool_call_id": c.id, "content": payload})
            if ok:
                evidence.append(result)
            trace.append({"step": step, "tool": c.function.name,
                          "arguments": c.function.arguments, "ok": ok,
                          "latency_s": round(time.time() - ts, 3),
                          "result_chars": len(payload),
                          "result_preview": payload[:4000]})
    if answer is None:
        answer = ("I couldn't finish within the step limit. Try a narrower question "
                  "(e.g. one store or one metric).")
    fc = check(answer, numbers_in(evidence) + numbers_in(question))
    record = {"ts": time.time(), "model": getattr(llm, "model", "?"), "question": question, "answer": answer, "tool_calls": trace,
              "steps": len(trace), "latency_s": round(time.time() - t0, 2),
              "numeric_accuracy": fc.accuracy, "unsupported_numbers": fc.unsupported,
              "supported_numbers": fc.supported}
    p = config.ROOT / cfg["trace_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    record["messages"] = messages + [{"role": "assistant", "content": answer}]
    return record


def main() -> None:
    if len(sys.argv) > 1:
        r = ask(" ".join(sys.argv[1:]))
        print(r["answer"])
        print(f"\n— {r['steps']} tool call(s), {r['latency_s']}s, numeric accuracy "
              f"{r['numeric_accuracy']:.0%}" + (f", unsupported: {r['unsupported_numbers']}"
                                                if r["unsupported_numbers"] else ""))
        return
    history: list[dict] = []
    print("Sales Insight Copilot — ask a question (blank line to quit)")
    while (q := input("\n> ").strip()):
        r = ask(q, history=history)
        print("\n" + r["answer"])
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]


if __name__ == "__main__":
    main()
