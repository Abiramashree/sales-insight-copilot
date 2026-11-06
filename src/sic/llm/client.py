"""LLM client for any OpenAI-compatible provider (Groq by default).

The provider is picked by `llm.provider` in config.yaml (or the SIC_LLM_PROVIDER env var).

Every call is appended to data/outputs/llm_calls.jsonl with purpose, latency,
token usage and cost, the raw material for the M5 eval/observability work.
`FakeLLM` implements the same interface with scripted replies for tests and
offline runs.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sic import config

_ORG = re.compile(r"\borg_[A-Za-z0-9]+")
_KEYISH = re.compile(r"\b(sk-[A-Za-z0-9-]{8,}|gsk_[A-Za-z0-9]{8,}|AQ\.[A-Za-z0-9_-]{12,})")


def redact(text: str) -> str:
    """Strip account identifiers / key-like strings from provider error messages."""
    return _KEYISH.sub("<redacted-key>", _ORG.sub("org_<redacted>", text))


class LLMConfigError(RuntimeError):
    pass


class LLMTruncatedError(RuntimeError):
    pass


def _log_path() -> Path:
    p = config.ROOT / config.load()["llm"]["log_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_LOG_LOCK = threading.Lock()


def _log(record: dict) -> None:
    with _LOG_LOCK, open(_log_path(), "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _dotenv() -> dict:
    """Minimal .env reader (project root, gitignored) so keys never live in code."""
    p = config.ROOT / ".env"
    if not p.exists():
        return {}
    pairs = (ln.split("=", 1) for ln in p.read_text().splitlines()
             if "=" in ln and not ln.lstrip().startswith("#"))
    return {k.strip(): v.strip().strip('"').strip("'") for k, v in pairs}


class LLM:
    """Thin wrapper: chat(messages, tools=None, json_mode=False, purpose='') -> message."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        base = config.load()["llm"]
        name = os.environ.get("SIC_LLM_PROVIDER") or base["provider"]
        if name not in base["providers"]:
            raise LLMConfigError(f"Unknown provider {name!r}; options: {list(base['providers'])}")
        # provider settings override the shared defaults
        cfg = {k: v for k, v in base.items() if k != "providers"} | base["providers"][name]
        cfg["provider"] = name
        key = os.environ.get(cfg["api_key_env"]) or _dotenv().get(cfg["api_key_env"])
        if not key:
            raise LLMConfigError(f"Set {cfg['api_key_env']} in your shell or in .env "
                                 f"(provider: {name}).")
        self.cfg = cfg
        self.model = model or cfg["model"]
        self.client = OpenAI(base_url=cfg["base_url"], api_key=key,
                             timeout=cfg["timeout_s"], max_retries=cfg["max_retries"],
                             default_headers=cfg.get("extra_headers"))

    def _create_with_rate_limit_wait(self, kwargs: dict, attempts: int = 6):
        """On 429, wait as long as the provider asks (retry-after header or the
        'try again in Xs' message), then retry. Free tiers cap tokens/minute."""
        from openai import RateLimitError

        for i in range(attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                if i == attempts - 1:
                    raise
                hdr = getattr(getattr(e, "response", None), "headers", {}) or {}
                m = re.search(r"(?:try again|retry) in (?:(\d+)h)?(?:(\d+)m(?!s))?([\d.]+)(ms|s)",
                              str(e), re.IGNORECASE)
                if m:
                    h, mins, secs, unit = m.groups()
                    parsed = (int(h or 0) * 3600 + int(mins or 0) * 60
                              + float(secs) / (1000 if unit == "ms" else 1))
                else:
                    parsed = 5.0
                wait = float(hdr.get("retry-after") or 0) or parsed
                if wait > 120:  # a daily/plan quota, not a per-minute limit: fail fast
                    raise
                time.sleep(wait + 0.5)
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, purpose: str = "", temperature: float | None = None,
             reasoning_effort: str | None = "config", model: str | None = None):
        kwargs: dict[str, Any] = {
            "model": model or self.model, "messages": messages,
            "temperature": self.cfg["temperature"] if temperature is None else temperature,
            "max_tokens": self.cfg["max_tokens"]}
        effort = self.cfg.get("reasoning_effort") if reasoning_effort == "config" \
            else reasoning_effort
        extra_body = dict(self.cfg.get("extra_body") or {})
        if effort:  # reasoning models: cap hidden "thinking" tokens
            kwargs["reasoning_effort"] = effort
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        error = None
        try:
            resp = self._create_with_rate_limit_wait(kwargs)
        except Exception as e:  # logged, then re-raised
            error = redact(f"{type(e).__name__}: {e}")
            raise
        finally:
            latency = time.time() - t0
            usage = getattr(resp, "usage", None) if error is None else None
            pin = getattr(usage, "prompt_tokens", 0) or 0
            pout = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None)
            choice = resp.choices[0] if error is None and resp.choices else None
            content = getattr(getattr(choice, "message", None), "content", None) or ""
            price = self.cfg["price_per_mtok"]
            _log({"ts": time.time(), "provider": self.cfg["provider"], "model": kwargs["model"],
                                "served_model": getattr(resp, "model", None) if error is None else None,
                  "purpose": purpose, "latency_s": round(latency, 3), "prompt_tokens": pin,
                  "completion_tokens": pout, "reasoning_tokens": reasoning,
                  "finish_reason": getattr(choice, "finish_reason", None),
                  "content_chars": len(content), "reasoning_effort": effort,
                  "cost_usd": round((pin * price["input"] + pout * price["output"]) / 1e6, 6),
                  "json_mode": json_mode, "n_tools": len(tools or []), "error": error})
        if choice and choice.finish_reason == "length" and not content and \
                not getattr(choice.message, "tool_calls", None):
            raise LLMTruncatedError(
                f"{purpose}: hit max_tokens={kwargs['max_tokens']} before any output "
                f"(reasoning_tokens={reasoning}). Lower llm.reasoning_effort or raise max_tokens.")
        return resp.choices[0].message


@dataclass
class FakeLLM:
    """Scripted stand-in. `replies` items are str (content) or dict with 'tool_calls':
    [{"name": ..., "arguments": {...}}]. Records every request for assertions."""

    replies: list[Any]
    requests: list[dict] = field(default_factory=list)
    model: str = "fake"

    def chat(self, messages, tools=None, json_mode=False, purpose="", temperature=None,
             reasoning_effort=None, model=None):
        self.requests.append({"messages": [dict(m) if isinstance(m, dict) else m
                                           for m in messages],
                              "tools": tools, "json_mode": json_mode, "purpose": purpose})
        r = self.replies.pop(0)
        if isinstance(r, str):
            return SimpleNamespace(content=r, tool_calls=None)
        calls = [SimpleNamespace(id=f"call_{i}", type="function",
                                 function=SimpleNamespace(name=c["name"],
                                                          arguments=json.dumps(c["arguments"])))
                 for i, c in enumerate(r["tool_calls"])]
        return SimpleNamespace(content=r.get("content"), tool_calls=calls)


def get_llm():
    return LLM()


def smoke() -> None:
    """`make llm-check`: one tiny call to verify key, base URL and model id."""
    llm = LLM()
    msg = llm.chat([{"role": "user", "content": "Reply with OK"}], purpose="smoke")
    print(f"{llm.model} via {llm.cfg['base_url']}: {msg.content!r}")


def probe() -> None:
    """`make llm-probe`: compare reasoning settings on a realistic card-sized task."""
    llm = LLM()
    task = [{"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Summarise in JSON {"headline": str, "bullets": [str]}: '
             "Store 921 sales were -23% vs forecast last week, driven by 21% fewer customer "
             "visits; no known cause."}]
    print(f"{'reasoning_effort':<18}{'latency':>9}{'out_tok':>9}{'reason_tok':>11}"
          f"{'finish':>9}  content")
    for effort in (None, "low", "medium"):
        t0 = time.time()
        try:
            msg = llm.chat(task, json_mode=True, purpose=f"probe:{effort}",
                           reasoning_effort=effort)
            rec = json.loads(_log_path().read_text().splitlines()[-1])
            print(f"{effort or 'default':<18}{time.time() - t0:>8.1f}s{rec['completion_tokens']:>9}"
                  f"{rec['reasoning_tokens']!s:>11}{rec['finish_reason']!s:>9}  "
                  f"{(msg.content or '')[:70]!r}")
        except Exception as e:  # noqa: BLE001 — probe reports any provider error
            print(f"{effort or 'default':<18}{time.time() - t0:>8.1f}s  ERROR {type(e).__name__}: "
                  f"{str(e)[:120]}")


if __name__ == "__main__":
    import sys

    probe() if "--probe" in sys.argv else smoke()
