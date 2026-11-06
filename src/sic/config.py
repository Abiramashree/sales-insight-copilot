"""Load config.yaml and resolve project paths."""

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


def path(key: str) -> Path:
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
