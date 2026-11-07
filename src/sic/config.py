"""Load config.yaml and resolve project paths.

Live feed: while data/live/feed.json exists (see sic.feed), the warehouse, outputs, reports
and figures resolve to data/live/ instead, and as_of() returns the last day loaded. Every
step and the app read paths through here, so they follow the feed with no other changes;
the full-data results under data/ and reports/ are never touched by a replay.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def load() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


LIVE = ROOT / "data" / "live"
LIVE_PATHS = {"warehouse": "warehouse.duckdb", "outputs": "outputs", "reports": "reports",
              "figures": "reports/figures"}


def feed_state() -> dict | None:
    """The live feed's state, or None when the app shows the full dataset."""
    f = LIVE / "feed.json"
    return json.loads(f.read_text()) if f.exists() else None


def as_of():
    """Last day of data loaded (a date), or None for the full dataset."""
    from datetime import date

    s = feed_state()
    return date.fromisoformat(s["as_of"]) if s else None


def path(key: str) -> Path:
    if key in LIVE_PATHS and feed_state() is not None:
        p = LIVE / LIVE_PATHS[key]
    else:
        p = Path(load()["paths"][key])
        p = p if p.is_absolute() else ROOT / p
    if key not in ("raw_source", "warehouse"):
        p.mkdir(parents=True, exist_ok=True)
    return p


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    wh = path("warehouse")
    wh.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(wh), read_only=read_only)


def read_jsonl(path: Path) -> list[dict]:
    """All records of a JSON-lines log ([] if missing)."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
