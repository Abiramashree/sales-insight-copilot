"""Live feed: replay the historical data into the warehouse one week at a time.

Rossmann is a historical dataset, so "live" is a replay. `start` opens a workspace in
data/live/ (config.path() redirects the warehouse, outputs and reports there) with data up
to the start date; `next` loads the following week and re-runs every step on the data
loaded so far. Nothing looks ahead:

  forecasts   precomputed by the rolling-origin backtest: each week's forecast comes from a
              model trained only on data before that quarter (the production retrain cycle)
  causal      re-estimated on every run from the data loaded so far
  everything  else (KPIs, anomalies, root cause, actions) is recomputed from the warehouse

`stop` deletes data/live/; the full-data results under data/ and reports/ are never touched.

    python -m sic.feed start [YYYY-MM-DD] | next | stop | status
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import date, timedelta

import pandas as pd

from sic import config

COPY_OUTPUTS = ["forecasts.parquet"]                      # produced by the quarterly backtest
COPY_REPORTS = ["m2_backtest_metrics.json", "m2_backtest_metrics_v1.json"]


def _cfg() -> dict:
    return config.load()["feed"]


def _write(state: dict) -> None:
    config.LIVE.mkdir(parents=True, exist_ok=True)
    tmp = config.LIVE / "feed.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, default=str))
    tmp.replace(config.LIVE / "feed.json")


def _count(sql: str):
    con = config.connect(read_only=True)
    try:
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()


# ------------------------------------------------------------------ steps
def _ingest(prev: date | None) -> str:
    from sic.data import build

    build.build()
    since = f"where date > date '{prev}'" if prev else ""
    return f"{_count(f'select count(*) from marts.fct_daily_store_sales {since}'):,} rows"


def _anomalies() -> str:
    from sic.detect import anomalies

    anomalies.run()
    w = pd.read_parquet(config.path("outputs") / "store_weeks.parquet")
    last = w[w.week_start == w.week_start.max()]
    return f"{int(last.is_anomaly.sum())} stores"


def _causal() -> str:
    from sic.causal import competitor_impact, promo2_did, promo_lift

    r = promo_lift.run()
    promo2_did.run()
    competitor_impact.run()
    return f"promo lift {r['pooled_lift'] * 100:+.0f}%"


def _root_cause() -> str:
    from sic.rca import engine

    engine.run()
    e = pd.read_parquet(config.path("outputs") / "episode_rca.parquet")
    return f"{int((e.root_cause != 'unexplained').sum())} of {len(e)} explained"


def _kpis() -> str:
    from sic.kpi import weekly

    weekly.run()
    return "updated"


def _actions() -> str:
    from sic.causal import recommend
    from sic.opportunity import score

    recommend.run()
    score.run(with_validation=False)
    return f"{len(pd.read_parquet(config.path('outputs') / 'recommendations.parquet'))} options"


STEPS = [("Ingest", _ingest), ("Anomalies", _anomalies), ("Causal", _causal),
         ("Root cause", _root_cause), ("KPIs", _kpis), ("Actions", _actions)]


def run_pipeline(prev: date | None = None) -> dict:
    """Run every step on the data loaded so far; log name, result and seconds per step."""
    state = config.feed_state()
    entry = {"as_of": state["as_of"], "ran_at": pd.Timestamp.now().isoformat(timespec="seconds"),
             "steps": []}
    for name, fn in STEPS:
        t0 = time.time()
        result = fn(prev) if name == "Ingest" else fn()
        entry["steps"].append({"step": name, "result": result,
                               "seconds": round(time.time() - t0, 1)})
    state["runs"] = (state.get("runs", []) + [entry])[-30:]
    _write(state)
    return entry


# ------------------------------------------------------------------ commands
def start(day: date | None = None) -> dict:
    """Open a fresh live workspace with data up to `day` (default feed.start)."""
    day = day or date.fromisoformat(_cfg()["start"])
    stop()                                         # paths below now resolve to full data
    src_out, src_rep = config.path("outputs"), config.path("reports")
    missing = [f for f in COPY_OUTPUTS if not (src_out / f).exists()]
    if missing:
        raise FileNotFoundError(f"Run `make forecast` first (missing {missing}).")
    _write({"as_of": day.isoformat(), "started": day.isoformat(), "runs": []})
    for f in COPY_OUTPUTS:
        shutil.copy2(src_out / f, config.path("outputs") / f)
    for f in COPY_REPORTS:
        if (src_rep / f).exists():
            shutil.copy2(src_rep / f, config.path("reports") / f)
    return run_pipeline()


def next_week() -> dict | None:
    """Load the next 7 days (up to the end of the data) and re-run. None at the end."""
    state = config.feed_state()
    if state is None:
        raise RuntimeError("The feed is not running. Start it first.")
    prev = date.fromisoformat(state["as_of"])
    end = date.fromisoformat(_cfg()["end"])
    if prev >= end:
        return None
    state["as_of"] = min(prev + timedelta(days=7), end).isoformat()
    _write(state)
    return run_pipeline(prev)


def stop() -> None:
    """Delete the live workspace; the app returns to the full dataset."""
    shutil.rmtree(config.LIVE, ignore_errors=True)


def status() -> dict | None:
    return config.feed_state()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        out = start(date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "next":
        out = next_week() or "end of data"
    elif cmd == "stop":
        out = stop() or "stopped"
    else:
        out = status() or "not running"
    print(json.dumps(out, indent=1, default=str))
