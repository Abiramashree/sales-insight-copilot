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


def test_feature_selection_bootstrap_and_redundancy():
    from sic.forecast import select as S

    rng = np.random.default_rng(0)
    n = 4000
    sales = rng.uniform(50, 150, n)
    w = {"store": rng.integers(0, 40, n), "sales": sales}
    a = sales * 1.05                       # 5% error everywhere
    b = sales * 1.10                       # 10% error everywhere -> delta exactly +5 pp
    lo, hi = S.bootstrap_delta([w], [a], [b], n=50)
    assert lo == pytest.approx(0.05, abs=1e-9) and hi == pytest.approx(0.05, abs=1e-9)

    x = rng.normal(size=n)
    X = pd.DataFrame({"a": x, "b": 2 * x + rng.normal(0, 0.01, n), "c": rng.normal(size=n),
                      "store_type": pd.Categorical(rng.choice(list("abcd"), n)),
                      "assortment": pd.Categorical(rng.choice(list("ab"), n)),
                      "state_holiday": pd.Categorical(["none"] * n)})
    r = S.redundancy(X, 0.8)
    assert ["a", "b"] in r["clusters"]
    assert "state_holiday" in r["near_constant"]


def test_feature_groups_cover_features_and_drop_list_validated(monkeypatch):
    from sic.forecast import features as F
    from sic.forecast import select as S

    assert sorted(f for g in S.GROUPS.values() for f in g) == sorted(F.FEATURES)
    cfg = {"forecast": {"drop_features": ["not_a_feature"]}}
    monkeypatch.setattr(F.config, "load", lambda: cfg)
    with pytest.raises(ValueError):
        F.active_features()


def test_weekly_kpis_compare_only_forecast_days_and_flag_partial_weeks():
    from sic import config

    p = config.path("outputs") / "weekly_kpis.parquet"
    if not p.exists():
        pytest.skip("run make kpis first")
    k = pd.read_parquet(p)
    chain = k[k.segment == "ALL"].sort_values("week_start")
    # first week starts before the first forecast day; it must not show a huge fake gap
    assert chain.sales_vs_forecast.abs().max() < 0.5
    assert bool(chain.partial_week.iloc[-1]) and (chain.days_in_data.iloc[-1] < 7)
    assert not chain.partial_week.iloc[:-1].any()
