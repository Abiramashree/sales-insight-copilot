"""LLM observability from data/outputs/llm_calls.jsonl (+ agent traces).

Per provider / model / call type: volume, latency p50/p95, tokens, error and
rate-limit rates, cost. Test runs with the fake LLM never reach this log.
"""

from __future__ import annotations

import json

import pandas as pd

from sic import config


def _kind(purpose: str) -> str:
    return purpose.split(":")[0] if purpose else "?"


def run() -> dict:
    p = config.ROOT / config.load()["llm"]["log_path"]
    df = pd.DataFrame(config.read_jsonl(p))
    if df.empty:
        return {"calls": 0}
    df["kind"] = df.purpose.map(_kind)
    df["is_error"] = df.error.notna()
    df["is_429"] = df.error.fillna("").str.contains("429")
    df["total_tokens"] = df.prompt_tokens.fillna(0) + df.completion_tokens.fillna(0)
    ok = df[~df.is_error]

    def agg(g: pd.DataFrame) -> pd.Series:
        s = g[~g.is_error]
        return pd.Series({
            "calls": len(g), "error_rate": g.is_error.mean(), "rate_limited": int(g.is_429.sum()),
            "latency_p50_s": s.latency_s.quantile(0.5) if len(s) else None,
            "latency_p95_s": s.latency_s.quantile(0.95) if len(s) else None,
            "avg_prompt_tokens": s.prompt_tokens.mean() if len(s) else None,
            "avg_completion_tokens": s.completion_tokens.mean() if len(s) else None,
            "truncated": int((s.finish_reason == "length").sum()) if "finish_reason" in s else 0,
            "cost_usd": g.cost_usd.sum(),
        })

    by_provider = df.groupby(["provider", "model"]).apply(agg, include_groups=False).reset_index()
    cur = config.load()["llm"]["provider"]
    by_kind = df[df.provider == cur].groupby(["kind", "model"]).apply(
        agg, include_groups=False).reset_index()
    tp = config.ROOT / config.load()["agent"]["trace_path"]
    traces = pd.DataFrame(config.read_jsonl(tp))
    if len(traces) and "model" in traces:
        traces = traces[traces.model != "fake"]
    result = {
        "calls": len(df), "ok_calls": len(ok),
        "current_provider": cur,
        "by_provider_model": by_provider.round(4).to_dict(orient="records"),
        "current_provider_by_call_type": by_kind.round(4).to_dict(orient="records"),
        "agent_runs": len(traces),
        "agent_mean_numeric_accuracy": float(traces.numeric_accuracy.mean()) if len(traces) else None,
        "agent_latency_p50_s": float(traces.latency_s.median()) if len(traces) else None,
    }
    (config.path("reports") / "m5_observability.json").write_text(
        json.dumps(result, indent=1, default=str))
    return result


if __name__ == "__main__":
    r = run()
    print(pd.DataFrame(r["by_provider_model"])[["provider", "model", "calls", "error_rate",
                                                 "rate_limited", "latency_p50_s",
                                                 "latency_p95_s", "truncated"]].to_string())
