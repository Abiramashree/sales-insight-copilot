"""The live replay loads data a week at a time, never looks ahead, and never touches the
full-data results. Runs the real pipeline (about 30 s, no LLM) in a temporary workspace."""

import hashlib
from datetime import date

import pandas as pd
import pytest

from sic import config, feed

pytestmark = pytest.mark.skipif(
    not (config.path("raw") / "train.csv").exists()
    or not (config.path("outputs") / "forecasts.parquet").exists(),
    reason="needs make data and make forecast")


def _fingerprint() -> str:
    h = hashlib.md5()
    for d in (config.path("outputs"), config.path("reports")):
        for p in sorted(d.rglob("*")):
            if p.is_file():
                h.update(p.read_bytes())
    return h.hexdigest()


def test_replay_week_by_week_without_lookahead():
    before = _fingerprint()                    # full-data outputs and reports
    feed.start(date(2015, 6, 28))
    assert config.as_of() == date(2015, 6, 28)
    assert feed.next_week()["as_of"] == "2015-07-05"
    out, cut = config.path("outputs"), pd.Timestamp("2015-07-05")
    assert str(out).startswith(str(config.LIVE))
    assert pd.read_parquet(out / "forecasts_calibrated.parquet").date.max() <= cut
    assert pd.read_parquet(out / "store_weeks.parquet").week_start.max() <= cut
    assert pd.read_parquet(out / "weekly_kpis.parquet").week_start.max() <= cut
    con = config.connect(read_only=True)
    assert con.execute("select max(date) from marts.fct_daily_store_sales").fetchone()[0] \
        == cut.date()
    con.close()
    steps = [s["step"] for s in config.feed_state()["runs"][-1]["steps"]]
    assert steps == [name for name, _ in feed.STEPS]
    feed.stop()
    assert config.feed_state() is None
    assert _fingerprint() == before
