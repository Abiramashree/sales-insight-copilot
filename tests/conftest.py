"""Tests always run against the full dataset, even if a live replay is open in data/live."""

import pytest

from sic import config


@pytest.fixture(autouse=True)
def _no_live_feed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIVE", tmp_path / "live")
