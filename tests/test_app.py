"""Every page renders without an exception. Uses pipeline outputs; no LLM calls
(chat and live cards only call the model on user action)."""

import pytest
from streamlit.testing.v1 import AppTest

from sic import config

PAGES = ["views/week.py", "views/stores.py", "views/ask.py"]
pytestmark = pytest.mark.skipif(
    not (config.path("outputs") / "weekly_kpis.parquet").exists(), reason="run make all first")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file("../app/main.py", default_timeout=60).run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_simulate_mode_runs_without_llm():
    at = AppTest.from_file("../app/main.py", default_timeout=60).run()
    at.switch_page("views/ask.py").run()
    at.segmented_control[0].set_value("Simulate").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Expected impact" for m in at.metric)
