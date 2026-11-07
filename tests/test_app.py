"""Every page renders without an exception and the no-LLM tools produce results. Uses
pipeline outputs; nothing here calls the LLM (cards and chat only do on user action)."""

import pytest
from streamlit.testing.v1 import AppTest

from sic import config

PAGES = ["views/pipeline.py", "views/actions.py", "views/test.py", "views/ask.py"]
pytestmark = pytest.mark.skipif(
    not (config.path("outputs") / "weekly_kpis.parquet").exists(), reason="run make all first")


def _app(page: str = PAGES[0]) -> AppTest:
    at = AppTest.from_file("../app/main.py", default_timeout=90).run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    _app(page)


def test_actions_store_view_runs_a_simulation():
    at = _app("views/actions.py")
    at.segmented_control[0].set_value("Stores").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Expected impact" for m in at.metric)


def test_tester_and_planner_return_numbers():
    at = _app("views/test.py")
    assert any(m.label == "Difference" and "95% CI" in m.delta for m in at.metric)
    at.segmented_control[0].set_value("Plan").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Detectable lift" for m in at.metric)


def test_ask_simulate_mode_runs_without_llm():
    at = _app("views/ask.py")
    at.segmented_control[0].set_value("Simulate").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Expected impact" for m in at.metric)
