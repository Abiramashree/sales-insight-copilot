"""What-if engine + agent tool tests (offline; FakeLLM, zero tokens)."""

import json

import pytest

from sic import config
from sic.llm.client import FakeLLM

needs_outputs = pytest.mark.skipif(
    not (config.ROOT / "data/outputs/promo_lift_by_store.parquet").exists(),
    reason="run `make all` first")


@needs_outputs
def test_promo_scenarios_are_signed_and_have_ordered_intervals():
    from sic.whatif.engine import run
    add = run(scenario="extra_promo_week", lowest_lift_n=20)
    rem = run(scenario="remove_promo_week", lowest_lift_n=20)
    t = add["total_eur"]
    assert 0 < t["p5"] < t["mean"] < t["p95"]
    assert rem["total_eur"]["mean"] == pytest.approx(-t["mean"], rel=0.02)
    two = run(scenario="extra_promo_week", lowest_lift_n=20, weeks=2)
    assert two["total_eur"]["mean"] == pytest.approx(2 * t["mean"], rel=0.02)


@needs_outputs
def test_competitor_near_hurts_more_than_far_and_uses_measured_effect():
    from sic.whatif.engine import run
    near = run(scenario="competitor_opens", store_ids=[262], competitor_distance_m=300)
    far = run(scenario="competitor_opens", store_ids=[262], competitor_distance_m=3000)
    rep = json.loads((config.path("reports") / "m3_competitor_impact.json").read_text())
    assert near["total_eur"]["mean"] < far["total_eur"]["mean"] < 0
    assert near["pct_of_baseline"]["mean"] == pytest.approx(
        rep["by_distance"]["within_1000m"]["effect_pct"], abs=0.01)


@needs_outputs
def test_promo2_is_low_confidence_and_skips_enrolled_stores():
    from sic.whatif.engine import _inputs, run
    dim = _inputs()["dim"]
    enrolled = int(dim[dim.promo2_enrolled].index[0])
    other = int(dim[~dim.promo2_enrolled].index[0])
    r = run(scenario="promo2_enroll", store_ids=[enrolled, other])
    assert r["confidence"] == "low" and r["stores"] == 1
    assert any("NOT proven" in c for c in r["caveats"])


@needs_outputs
@pytest.mark.parametrize("kw", [dict(scenario="price_cut", all_stores=True),
                                dict(scenario="competitor_opens", store_ids=[262]),
                                dict(scenario="extra_promo_week", store_ids=[99999]),
                                dict(scenario="extra_promo_week")])
def test_invalid_scenarios_are_rejected(kw):
    from sic.whatif.engine import ScenarioError, run
    with pytest.raises(ScenarioError):
        run(**kw)


@needs_outputs
def test_agent_simulates_with_nulls_and_trace_supports_show_your_work():
    from sic.agent.chat import ask
    from sic.whatif.engine import run
    expected = run(scenario="competitor_opens", store_ids=[262], competitor_distance_m=300)
    k = round(abs(expected["total_eur"]["mean"]) / 1000)
    llm = FakeLLM([
        {"tool_calls": [{"name": "simulate_scenario", "arguments": {
            "scenario": "competitor_opens", "store_ids": [262], "store_type": None,
            "lowest_lift_n": None, "weakest_n": None, "all_stores": None, "weeks": None,
            "competitor_distance_m": 300}}]},
        f"Store 262 would lose about €{k}k over 26 weeks (high confidence).",
    ])
    r = ask("What if a competitor opens 300 m from store 262?", llm=llm)
    call = r["tool_calls"][0]
    assert call["tool"] == "simulate_scenario" and call["ok"]
    assert json.loads(call["result_preview"])["confidence"] == "high"
    assert r["numeric_accuracy"] == 1.0 and f"€{k}k" in r["supported_numbers"]
