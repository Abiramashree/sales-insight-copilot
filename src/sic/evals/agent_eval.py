"""Agent eval on evals/agent_golden.yaml: correctness, tool choice, numeric accuracy, latency.

Live LLM required. Writes reports/m5_agent_eval.json.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import yaml

from sic import config
from sic.agent import tools as T
from sic.insights.factcheck import check, normalize
from sic.llm.client import redact


def _scalar(sql: str):
    r = T.sandbox().execute(sql).fetchone()
    return r[0] if r else None


def run_check(answer: str, c: dict) -> tuple[bool, str]:
    answer = re.sub(r"[\u2010\u2011\u2012]", "-", normalize(answer))  # PB‑COMP (nb hyphen)
    low = answer.lower()
    if "contains_any" in c:
        ok = any(w.lower() in low for w in c["contains_any"])
        return ok, f"contains one of {c['contains_any'][:3]}…"
    if "value" in c:
        v = float(_scalar(c["value"]))
        # fact-checker accepts 0.226 written as 22.6% / 23%, money rounding, exact counts
        ok = bool(check(answer, [v], allow_small=False).supported)
        return ok, f"value {v:.4g} ({c['kind']})"
    if "report" in c:  # expected value read from a pipeline report (e.g. causal estimate)
        node = json.loads((config.path("reports") / c["report"]).read_text())
        for key in c["path"].split("."):
            node = node[key]
        v = float(node)
        ok = bool(check(answer, [v], allow_small=False).supported)
        return ok, f"report value {v:.4g} ({c['report']})"
    if "text" in c:
        v = _scalar(c["text"])
        pat = c.get("pattern", r"\b{v}\b").replace("{v}", re.escape(str(v)))
        ok = re.search(pat, answer, re.IGNORECASE) is not None
        return ok, f"text {v!r}"
    raise ValueError(c)


def run(ids: list[str] | None = None, out_name: str = "m5_agent_eval.json",
        model: str | None = None) -> dict:
    from sic.agent.chat import ask
    from sic.llm.client import LLM

    gold = yaml.safe_load((config.ROOT / "evals" / "agent_golden.yaml").read_text())
    if ids:
        gold = [g for g in gold if g["id"] in ids or any(g["id"].startswith(i) for i in ids)]
    llm = LLM(model=model)
    log = config.ROOT / config.load()["llm"]["log_path"]
    n_before = len(config.read_jsonl(log))
    rows = []
    for g in gold:
        t0 = time.time()
        try:
            r = ask(g["q"], llm=llm)
            err = None
        except Exception as e:  # noqa: BLE001 - record and continue
            r, err = {"answer": "", "tool_calls": [], "numeric_accuracy": 0.0,
                      "unsupported_numbers": [], "steps": 0}, redact(f"{type(e).__name__}: {e}")[:200]
        used = [c["tool"] for c in r["tool_calls"]]
        checks = [run_check(r["answer"], c) for c in g["checks"]]
        rows.append({
            "id": g["id"], "q": g["q"], "answer": r["answer"], "error": err,
            "correct": err is None and all(ok for ok, _ in checks),
            "checks": [{"ok": ok, "what": what} for ok, what in checks],
            "tools_used": used,
            "tool_ok": (not g["tools"]) or any(t in used for t in g["tools"]),
            "numeric_accuracy": r["numeric_accuracy"],
            "unsupported_numbers": r["unsupported_numbers"],
            "steps": r["steps"], "latency_s": round(time.time() - t0, 2),
        })
        print(f"[{'ERR ' if err else 'PASS' if rows[-1]['correct'] else 'FAIL'}] {g['id']:<22} "
              f"{rows[-1]['latency_s']:>6.1f}s tools={used}")
    calls = config.read_jsonl(log)[n_before:]
    scored = [r for r in rows if r["error"] is None]  # infra errors are not wrong answers
    lat = np.array([r["latency_s"] for r in scored] or [0.0])
    result = {
        "n": len(rows),
        "infra_errors": len(rows) - len(scored),
        "correctness": float(np.mean([r["correct"] for r in scored])) if scored else None,
        "tool_selection_accuracy": float(np.mean([r["tool_ok"] for r in scored])) if scored else None,
        "mean_numeric_accuracy": float(np.mean([r["numeric_accuracy"] for r in scored])) if scored else None,
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "mean_tool_calls": float(np.mean([r["steps"] for r in scored])) if scored else None,
        "llm_calls": len(calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "rate_limit_errors_logged": sum(1 for c in calls if c.get("error") and "429" in c["error"]),
        "model": getattr(llm, "model", "?"), "provider": llm.cfg.get("provider"),
        "rows": rows,
    }
    (config.path("reports") / out_name).write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    ids = args[args.index("--ids") + 1].split(",") if "--ids" in args else None
    out = args[args.index("--out") + 1] if "--out" in args else "m5_agent_eval.json"
    mdl = args[args.index("--model") + 1] if "--model" in args else None
    r = run(ids, out, mdl)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
