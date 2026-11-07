"""Playbook RAG: governed retrieval of approved store-response playbooks.

Retrieval rules (enforced before anything reaches an LLM):
  1. status == approved             (drafts and superseded versions are never returned)
  2. effective_from <= as_of <= review_by   (expired guidance is never returned)
  3. latest eligible version per playbook id
Then either:
  match(...)   structured: root cause / driver / direction / competitor distance from the
               RCA evidence (used for insight cards and store questions)
  search(...)  semantic: dense embeddings over title + situation (free-text questions)
Expected-impact text is rendered from the causal analysis report, so quoted numbers
always equal the latest estimate.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

import pandas as pd
import yaml

from sic import config

PATH = config.ROOT / "docs" / "playbooks" / "playbooks.yaml"
AS_OF = date(2015, 7, 31)  # last day of the full data; the live feed's as-of date overrides it


def as_of() -> date:
    """Governance is evaluated as of the last day of data loaded."""
    return config.as_of() or AS_OF


def load_all() -> list[dict]:
    return yaml.safe_load(PATH.read_text())


def eligible(day: date | None = None, books: list[dict] | None = None) -> list[dict]:
    day = day or as_of()
    books = load_all() if books is None else books
    ok = [b for b in books if b["status"] == "approved"
          and b["effective_from"] <= day <= b["review_by"]]
    latest: dict[str, dict] = {}
    for b in ok:
        if b["id"] not in latest or b["version"] > latest[b["id"]]["version"]:
            latest[b["id"]] = b
    return list(latest.values())


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _impact_values() -> dict:
    return _load_impact_values(str(config.as_of()))


@lru_cache(maxsize=2)
def _load_impact_values(data_key: str) -> dict:
    p = config.path("reports") / "m3_competitor_impact.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    near, far = r["by_distance"]["within_1000m"], r["by_distance"]["beyond_1000m"]
    return {"near_effect": _pct(near["effect_pct"]), "near_lo": _pct(near["ci_pct"][0]),
            "near_hi": _pct(near["ci_pct"][1]), "near_n": near["treated_stores"],
            "far_effect": _pct(far["effect_pct"]), "far_lo": _pct(far["ci_pct"][0]),
            "far_hi": _pct(far["ci_pct"][1]), "far_n": far["treated_stores"],
            "credible": r.get("credible", False)}


def render(book: dict) -> dict:
    vals = _impact_values()
    impact = " ".join(str(book.get("expected_impact", "")).split())
    if "{" in impact:
        impact = impact.format(**vals) if vals else "Impact analysis not available."
    return {
        "citation": f"{book['id']} v{book['version']}",
        "title": book["title"],
        "situation": " ".join(str(book["situation"]).split()),
        "expected_impact": impact,
        "plays": [f"{p['step']} ({p['owner']}, {p['when']})" for p in book["plays"]],
        "talking_points": book.get("talking_points", []),
        "measure": book.get("measure", []),
        "do_not": book.get("do_not", []),
        "owner": book["owner"], "approved_by": book["approved_by"],
        "review_by": str(book["review_by"]),
    }


def competitor_distance(evidence: list[dict]) -> float | None:
    for ev in evidence:
        if ev.get("type") == "competitor_opened":
            m = re.search(r"([\d.]+)\s*m away", ev.get("detail", ""))
            if m:
                return float(m.group(1))
    return None


def match(root_cause: str | None, driver: str | None, direction: int | None,
          competitor_m: float | None = None, k: int = 2) -> list[dict]:
    """Structured match against RCA output. Returns rendered playbooks, best first."""
    scored = []
    for b in eligible():
        a = b.get("applies_when") or {}
        score = 0
        if "root_cause" in a:
            if root_cause not in a["root_cause"]:
                continue
            score += 3
        if "driver" in a:
            if driver not in a["driver"]:
                continue
            score += 1
        if "direction" in a:
            if direction != a["direction"]:
                continue
            score += 1
        if "competitor_within_m" in a and not (competitor_m is not None
                                               and competitor_m <= a["competitor_within_m"]):
            continue
        if "competitor_beyond_m" in a and not (competitor_m is not None
                                               and competitor_m > a["competitor_beyond_m"]):
            continue
        scored.append((score, b))
    scored.sort(key=lambda x: -x[0])
    return [render(b) for _, b in scored[:k]]


def for_episode(ep: pd.Series | dict, k: int = 2) -> list[dict]:
    ev = json.loads(ep["evidence"]) if isinstance(ep["evidence"], str) else ep["evidence"]
    return match(ep["root_cause"], ep.get("tb_driver"), int(ep["direction"]),
                 competitor_distance(ev), k=k)


def for_store(store_id: int, k: int = 2) -> dict:
    e = pd.read_parquet(config.path("outputs") / "episode_rca.parquet")
    e = e[e.store_id == store_id].sort_values("end")
    if e.empty:
        return {"store_id": store_id, "episode": None, "playbooks": [],
                "note": "No anomaly episode for this store; no playbook triggered."}
    ep = e.iloc[-1]
    return {"store_id": store_id,
            "episode": {"start": str(ep.start.date()), "end": str(ep.end.date()),
                        "gap_pct": round(float(ep.gap_pct), 3), "root_cause": ep.root_cause,
                        "driver": ep.tb_driver},
            "playbooks": for_episode(ep, k=k)}


@lru_cache
def _index():
    import numpy as np

    from sic.agent.retrieval import embed_model

    books = eligible()
    texts = [f"{b['title']}. {' '.join(str(b['situation']).split())} "
             f"Triggers: {json.dumps(b.get('applies_when', {}))}" for b in books]
    model = embed_model()
    if model is None:
        return books, None, None
    E = np.array(list(model.embed(texts)))
    return books, model, E / np.linalg.norm(E, axis=1, keepdims=True)


_DIST = re.compile(r"(\d+(?:[.,]\d+)?)\s*(km|kilomet\w*|m\b|met(?:er|re)s?\b)", re.IGNORECASE)
_COMP = re.compile(r"competit|rival|opened|opening|new store|chain", re.IGNORECASE)


def distance_in(query: str) -> float | None:
    """Explicit distance in a question ('200 m', '3 km') -> metres. Embeddings can't
    compare numbers against a 1 km threshold, so this is applied as a rule."""
    m = _DIST.search(query)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v * 1000 if m.group(2).lower().startswith("k") else v


def _distance_ok(book: dict, metres: float) -> bool:
    a = book.get("applies_when") or {}
    if "competitor_within_m" in a and metres > a["competitor_within_m"]:
        return False
    return not ("competitor_beyond_m" in a and metres <= a["competitor_beyond_m"])


def search(query: str, k: int = 2) -> list[dict]:
    """Semantic retrieval over ELIGIBLE playbooks only (+ distance rule if stated)."""
    import numpy as np

    from sic.agent.retrieval import QUERY_PREFIX

    books, model, E = _index()
    metres = distance_in(query) if _COMP.search(query) else None
    if metres is not None:
        keep = [i for i, b in enumerate(books) if _distance_ok(b, metres)]
        books = [books[i] for i in keep]
        E = E[keep] if E is not None else None
    if model is None:  # lexical fallback
        q = set(query.lower().split())
        ranked = sorted(books, key=lambda b: -len(q & set(f"{b['title']} {b['situation']}"
                                                         .lower().split())))
        return [render(b) for b in ranked[:k]]
    v = np.array(list(model.embed([QUERY_PREFIX + query])))[0]
    sims = E @ (v / np.linalg.norm(v))
    return [render(books[i]) | {"score": round(float(sims[i]), 3)}
            for i in sims.argsort()[::-1][:k]]
