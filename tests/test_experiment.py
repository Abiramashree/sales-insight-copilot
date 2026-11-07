"""Experiment design, readout and planner: reproducible, balanced, and they recover known
effects. Uses the warehouse; no LLM."""

import numpy as np
import pandas as pd
import pytest

from sic import config
from sic import experiment as X
from sic import planner as P

pytestmark = pytest.mark.skipif(
    not config.path("warehouse").exists()
    or not (config.path("outputs") / "opportunity_scores.parquet").exists(),
    reason="needs make all")
STORES = list(range(1, 81))


def test_assignment_is_reproducible_and_balanced():
    a, b = X.assign(STORES, seed=3), X.assign(STORES, seed=3)
    assert a.equals(b)
    assert not a.equals(X.assign(STORES, seed=4))
    n = a.group.value_counts()
    assert abs(n["test"] - n["control"]) <= 1
    m = a.groupby("group").weekly_sales.mean()
    assert abs(m["test"] / m["control"] - 1) < 0.10       # stratified pairs keep groups alike


def test_dry_run_is_an_honest_aa_test():
    r = X.dry_run(X.assign(STORES), "2015-06-01", 6)
    assert r["dry_run"] and r["cuped"]
    assert r["cuped"]["ci"][0] < 0 < r["cuped"]["ci"][1]
    assert r["decision"] == "do not roll out"


def test_readout_recovers_an_injected_lift():
    groups = X.assign(STORES)
    w = X.weekly()
    span = sorted(w[w.week_start >= pd.Timestamp("2015-06-01")].week_start.unique())[:6]
    res = w[w.week_start.isin(span)].merge(groups[["store_id", "group"]], on="store_id")
    res.loc[res.group == "test", "sales"] *= 1.05
    r = X.analyze(res[X.REQUIRED], planned_lift=0.05)
    assert r["cuped"]["ci"][0] < 0.05 < r["cuped"]["ci"][1]
    assert r["cuped"]["lift"] == pytest.approx(0.05, abs=0.02)
    # CUPED narrows the interval vs the raw comparison
    width = {k: r[k]["ci"][1] - r[k]["ci"][0] for k in ("raw", "cuped")}
    assert width["cuped"] < width["raw"]


def test_csv_validation():
    ok = X.read_csv("store_id,week_start,group,sales\n1,2015-06-01,test,100\n2,2015-06-01,Control,90\n")
    assert list(ok.group) == ["test", "control"]
    with pytest.raises(ValueError, match="Missing columns"):
        X.read_csv("store,week,sales\n1,2015-06-01,100\n")
    with pytest.raises(ValueError, match="test' or 'control"):
        X.read_csv("store_id,week_start,group,sales\n1,2015-06-01,A,100\n")
    with pytest.raises(ValueError, match="both groups"):
        X.read_csv("store_id,week_start,group,sales\n1,2015-06-01,test,1\n1,2015-06-08,control,1\n")


def test_planner_is_deterministic_and_consistent():
    a, b = P.plan(0.02), P.plan(0.02)
    assert a["stores"] == b["stores"] and a["experiment"]["groups"] == b["experiment"]["groups"]
    pl = a["plausibility"]
    assert pl["verdict"] == P._verdict(pl["coverage"])
    if pl["coverage"] >= 1:                       # enough upside -> the stores cover the goal
        assert a["gap_eur_per_week"] == pytest.approx(0, abs=1)
    assert sum(s["add_eur_per_week"] for s in a["stores"]) == pytest.approx(
        a["covered_eur_per_week"], rel=0.01)
    assert all(np.isfinite(x["eur_per_week"]) for x in a["alternatives"])
    with pytest.raises(ValueError):
        P.plan(5)                                 # must be a fraction
    with pytest.raises(ValueError, match="Unknown stores"):
        P.plan(0.05, store_ids=[99999])
    assert "note" in P.plan(0.1, store_ids=[262, 770, 1114])["experiment"]
