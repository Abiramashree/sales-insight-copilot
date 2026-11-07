"""Demo mode: live insight cards when the LLM is healthy, saved cards when it isn't.

  make demo            try live (quick health check + time budget), else show snapshot
  make demo-cached     show the saved snapshot immediately (no network)

The snapshot (demo/insight_cards_snapshot.json) is refreshed only by a run where at
least `snapshot_min_publish_rate` of cards passed QA, so a rate-limited or broken run
can never overwrite good demo content.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from datetime import UTC, datetime

from sic import config

SNAPSHOT = config.ROOT / "demo" / "insight_cards_snapshot.json"
SNAPSHOT_MD = config.ROOT / "demo" / "insight_cards_snapshot.md"
_STRIP = re.compile(r"\s*\[f\d+(?:\s*,\s*f\d+)*\]")


def _cfg() -> dict:
    return config.load()["insights"]


def save_snapshot(cards: list[dict], provider: str, model: str) -> bool:
    """Persist a good run as the demo snapshot. Returns True if saved. Never during a live
    replay: the snapshot is the full-history briefing."""
    if config.feed_state():
        print("Snapshot not updated: live feed active (cards are for a replayed week).")
        return False
    from sic.insights.cards import render_md

    rate = sum(c["qa"]["published"] for c in cards) / max(len(cards), 1)
    if rate < _cfg().get("snapshot_min_publish_rate", 0.8):
        print(f"Snapshot NOT updated: only {rate:.0%} of cards passed QA "
              f"(need {_cfg().get('snapshot_min_publish_rate', 0.8):.0%}).")
        return False
    SNAPSHOT.parent.mkdir(exist_ok=True)
    meta = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": provider, "model": model, "publish_rate": rate,
            "data_week": next((c["subject"].split("week of ")[-1] for c in cards
                               if c["kind"] == "chain_weekly"), None)}
    keep = [{k: c[k] for k in ("subject", "kind", "card", "qa")} for c in cards]
    SNAPSHOT.write_text(json.dumps({"meta": meta, "cards": keep}, indent=1, default=str))
    SNAPSHOT_MD.write_text(render_md(cards).replace(
        "# Weekly insight cards", f"# Weekly insight cards (saved demo · {meta['model']} · "
        f"{meta['generated_at'][:10]})", 1))
    print(f"Snapshot saved: demo/{SNAPSHOT.name} ({rate:.0%} published)")
    return True


def load_snapshot() -> dict | None:
    return json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else None


def print_cards(cards: list[dict], only_published: bool = True) -> None:
    for c in cards:
        if only_published and not c["qa"]["published"]:
            continue
        k = c["card"]
        print(f"\n■ {_STRIP.sub('', k.get('title', c['subject']))}   "
              f"[{c['subject']} · confidence {k.get('confidence', '?')}]")
        print(f"  {_STRIP.sub('', k.get('headline', ''))}")
        for b in k.get("bullets", []):
            print(f"   • {_STRIP.sub('', b)}")
        print(f"  → {_STRIP.sub('', k.get('action', ''))}")


def show_cached(reason: str | None = None) -> int:
    snap = load_snapshot()
    if reason:
        print(f"\n⚠  Live LLM unavailable ({reason}).")
    if not snap:
        print("No saved demo cards yet. Run `make cards` once while the LLM is healthy.")
        return 1
    m = snap["meta"]
    print(f"▶  Showing saved demo cards — generated {m['generated_at'][:16].replace('T', ' ')} "
          f"UTC with {m['model']} ({m['provider']}), data week {m['data_week']}, "
          f"{m['publish_rate']:.0%} passed QA.")
    print_cards(snap["cards"])
    return 0


def health_check(timeout_s: float) -> str | None:
    """Return None if healthy, else a short reason."""
    from sic.llm.client import LLM, LLMConfigError

    try:
        llm = LLM()
        llm.client = llm.client.with_options(timeout=timeout_s, max_retries=0)
        t0 = time.time()
        llm.chat([{"role": "user", "content": "Reply with OK"}], purpose="demo_health")
        if time.time() - t0 > timeout_s:
            return f"slow: health check took {time.time() - t0:.0f}s"
        return None
    except LLMConfigError as e:
        return f"not configured: {e}"
    except Exception as e:  # noqa: BLE001 - any failure means fall back
        from sic.llm.client import redact
        return redact(f"{type(e).__name__}: {str(e)[:80]}")


def live_or_cached() -> int:
    cfg = _cfg()
    reason = health_check(cfg.get("health_check_timeout_s", 10))
    if reason:
        return show_cached(reason)

    from sic.insights import cards as C

    result: dict = {}

    def work():
        try:
            result["cards"] = C.run()
        except Exception as e:  # noqa: BLE001
            result["error"] = f"{type(e).__name__}: {str(e)[:80]}"

    budget = cfg.get("live_time_budget_s", 240)
    print(f"Generating live cards (time budget {budget}s)…")
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(budget)
    if t.is_alive():
        return show_cached(f"live generation exceeded {budget}s")
    if "error" in result:
        return show_cached(result["error"])
    cards = result["cards"]
    rate = sum(c["qa"]["published"] for c in cards) / max(len(cards), 1)
    if rate < cfg.get("snapshot_min_publish_rate", 0.8):
        return show_cached(f"live run degraded: only {rate:.0%} of cards passed QA")
    print(f"▶  Live cards ({rate:.0%} passed QA):")
    print_cards(cards)
    return 0


if __name__ == "__main__":
    sys.exit(show_cached() if "--cached" in sys.argv else live_or_cached())
