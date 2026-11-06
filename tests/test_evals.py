"""Eval regression gates: fail the build if QA or retrieval quality drops."""

import pytest

from sic import config

needs_outputs = pytest.mark.skipif(
    not (config.ROOT / "data/outputs/recommendations.parquet").exists(),
    reason="run `make all` first")

RULE_DEFECTS = ["sign_flip", "store_id_typo", "double_negative", "at_risk", "missing_hedge",
                "wow_lead"]


@needs_outputs
def test_card_qa_gate_catches_rule_defects_without_false_alarms():
    from sic.evals.card_qa_eval import run
    r = run(use_llm=False)
    assert r["deterministic_false_positive_rate"] == 0.0, r["clean_false_positives"]
    for d in RULE_DEFECTS:
        if d in r["by_defect"]:
            assert r["by_defect"][d]["deterministic"] == 1.0, d
    assert r["by_defect"]["wrong_number"]["deterministic"] >= 0.8


def test_retrieval_quality_floor():
    from sic.evals.retrieval_eval import score
    dev = score("retrieval_golden.yaml", "dense")
    hold = score("retrieval_holdout.yaml", "dense")
    assert dev["recall@3"] >= 0.85 and hold["recall@1"] >= 0.5 and hold["recall@3"] >= 0.6
