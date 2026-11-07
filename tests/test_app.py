"""Both pages render, the KPI drill-down works for chain and store, the assistant is a
single chat, and an anomaly hands off to it. The LLM is stubbed: no tokens."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from sic import config

pytestmark = pytest.mark.skipif(
    not (config.path("outputs") / "weekly_kpis.parquet").exists(), reason="run make all first")


def _app(page: str | None = None) -> AppTest:
    at = AppTest.from_file("../app/main.py", default_timeout=120).run()
    if page:
        at.switch_page(page).run()
    _ok(at)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _labels(at: AppTest) -> list[str]:
    return [m.label for m in at.metric]


def test_kpis_chain_week_month_quarter():
    at = _app()
    assert _labels(at)[:3] == ["Sales", "vs forecast", "vs last year"]
    assert "Gap vs forecast" in _labels(at)            # anomaly drill-down rendered
    for grain in ("month", "quarter"):
        at.segmented_control[0].set_value(grain).run()
        _ok(at)
        assert at.metric[0].value not in ("—", "")


def test_kpis_single_store():
    at = _app()
    at.selectbox[0].set_value(1114).run()
    _ok(at)
    assert at.metric[0].label == "Sales" and at.metric[0].value.startswith("€")


def test_assistant_is_one_chat():
    at = _app("views/assistant.py")
    assert len(at.chat_input) == 1 and not at.segmented_control      # no manual panel
    assert at.get("button_group")                                     # example prompts


def test_anomaly_hands_off_to_assistant(monkeypatch):
    seen = {}

    def fake_ask(q, history=None, **kw):
        seen["q"] = q
        args = json.dumps({"target_pct": 0.05, "store_type": "b"})
        return {"answer": "**Goal** stub", "unsupported_numbers": [],
                "tool_calls": [{"tool": "plan_initiative", "arguments": args, "ok": True,
                                "latency_s": 0.1, "result_preview": "{}"}]}

    import sic.agent.chat
    monkeypatch.setattr(sic.agent.chat, "ask", fake_ask)
    at = _app()
    next(b for b in at.button if b.label == "Investigate with the assistant").click().run()
    _ok(at)
    assert seen["q"].startswith("Investigate store ")
    assert any(m.label == "Weekly sales now" for m in at.metric)   # plan block rendered
