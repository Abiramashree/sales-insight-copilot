"""Store opportunity scoring tests (offline, zero tokens)."""

import json

import numpy as np
import pandas as pd
import pytest

from sic import config
from sic.llm.client import FakeLLM

needs_outputs = pytest.mark.skipif(
    not (config.ROOT / "data/outputs/opportunity_scores.parquet").exists(),
    reason="run `uv run python -m sic.opportunity.score` first")


@needs_outputs
def test_scores_are_consistent_with_levers_and_weights():
    o = pd.read_parquet(config.path("outputs") / "opportunity_scores.parquet")
    w = config.load()["opportunity"]["confidence_weights"]
    recomputed = o.eur_peer * w["peer"] + o.eur_promo * w["promo"] + o.eur_competitor * w["competitor"]
    assert np.allclose(o.score_eur, recomputed)
    assert o.score_eur.notna().all() and (o[["eur_peer", "eur_promo", "eur_competitor"]] >= 0).all().all()
    assert list(o["rank"]) == list(range(1, len(o) + 1))
    assert o.score_eur.is_monotonic_decreasing
    assert len(o) == o.store_id.nunique()


@needs_outputs
def test_competitor_lever_only_for_recent_competitors_within_5km():
    o = pd.read_parquet(config.path("outputs") / "opportunity_scores.parquet")
    has = o[o.eur_competitor > 0]
    assert has.competitor_recent.all() and (has.competition_distance_m <= 5000).all()


@needs_outputs
def test_validation_shows_levers_are_persistent_and_basket_not_size():
    v = json.loads((config.path("reports") / "m3_opportunity.json").read_text())["validation"]
    assert v["promo_lift_split_sample_spearman"] > 0.8
    assert v["basket_gap_year_over_year_spearman"] > 0.8
    assert abs(v["basket_gap_vs_store_size_spearman"]) < 0.5


@needs_outputs
def test_agent_prioritizes_via_tool():
    from sic.agent.chat import ask
    o = pd.read_parquet(config.path("outputs") / "opportunity_scores.parquet")
    top = int(o.iloc[0].store_id)
    llm = FakeLLM([
        {"tool_calls": [{"name": "get_opportunities", "arguments": {
            "limit": 5, "opportunity_type": None, "store_type": None, "store_id": None}}]},
        f"Start with store {top}, the top-ranked opportunity.",
    ])
    r = ask("Which stores should we prioritize this quarter?", llm=llm)
    c = r["tool_calls"][0]
    assert c["tool"] == "get_opportunities" and c["ok"]
    assert json.loads(c["result_preview"])["opportunities"][0]["store_id"] == top
