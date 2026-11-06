"""Unit tests for the math the pipeline's claims rest on."""

import numpy as np
import pandas as pd
import pytest

from sic.causal.fe import fe_ols
from sic.detect.anomalies import episodes
from sic.rca.engine import drill_down, traffic_basket


def test_fe_ols_recovers_known_effect_with_two_way_fe():
    rng = np.random.default_rng(0)
    units, periods, true = 200, 40, 0.10
    df = pd.DataFrame([(u, t) for u in range(units) for t in range(periods)],
                      columns=["unit", "t"])
    unit_fe = rng.normal(0, 1, units)[df.unit]
    time_fe = rng.normal(0, 0.5, periods)[df.t]
    # staggered, unit-specific starts; treatment correlated with unit FE (confounded
    # in a naive regression, but absorbed by the unit fixed effect)
    start = np.where(unit_fe > 0, 20, 99)[df.unit]
    df["d"] = (df.t >= start).astype(float)
    df["y"] = unit_fe + time_fe + true * df.d + rng.normal(0, 0.1, len(df))
    df = df.sample(frac=0.9, random_state=1)  # unbalanced panel
    r = fe_ols(df, "y", ["d"], ["unit", "t"], "unit")
    assert r.params["d"] == pytest.approx(true, abs=0.01)
    assert r.ci().loc["d", "lo"] < true < r.ci().loc["d", "hi"]


def test_traffic_basket_decomposition_is_exact():
    tb = traffic_basket(sales=90, forecast=100, customers=80, forecast_customers=100)
    assert tb["traffic_log_gap"] + tb["basket_log_gap"] == pytest.approx(tb["total_log_gap"])
    assert tb["driver"] == "traffic" or tb["traffic_share"] > 1  # traffic fell more than sales


def test_drill_down_contributions_cover_the_gap():
    rows = pd.DataFrame({"seg": ["a", "a", "b", "c"], "sales": [110, 120, 100, 100],
                         "forecast": [100, 100, 100, 100]})
    dd = drill_down(rows, ["seg"])
    elems = {e["value"]: e for e in dd[0]["elements"]}
    assert elems["a"]["contribution"] == pytest.approx(1.0)  # all of the +30 gap
    assert "b" not in elems  # zero-gap segments are not reported as causes


def test_episodes_merge_consecutive_same_direction_weeks_only():
    wk = pd.to_datetime(["2015-01-05", "2015-01-12", "2015-01-19", "2015-02-02"])
    w = pd.DataFrame({
        "store_id": 1, "store_type": "a", "week_start": wk, "is_anomaly": True,
        "direction": [1, 1, -1, -1], "sales": 100.0, "forecast": 80.0,
        "customers": 10.0, "forecast_customers": 8.0,
    })
    e = episodes(w)
    assert len(e) == 3  # [w1,w2] up, [w3] down, [w5] down after a gap
    assert sorted(e.weeks.tolist()) == [1, 1, 2]
