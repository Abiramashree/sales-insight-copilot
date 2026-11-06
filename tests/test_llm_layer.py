"""M4 tests: fact-check, SQL sandbox, agent loop and card QA, all with a fake LLM."""

import json

import pytest

from sic import config
from sic.insights.factcheck import check, extract_numbers
from sic.llm.client import FakeLLM

needs_outputs = pytest.mark.skipif(
    not (config.ROOT / "data/outputs/recommendations.parquet").exists(),
    reason="run `make all` first")


# ---------- fact-check ----------
def test_factcheck_accepts_faithful_renderings():
    src = [-0.226, -13326.0, 0.92, 921]
    fc = check("Store 921 is down 23% (-€13.3k) [f3, f4]; visits explain 92% [f10].", src)
    assert fc.passed, fc.unsupported


def test_factcheck_rejects_invented_and_sign_flipped_numbers():
    src = [-0.226, 0.92]
    assert check("Sales fell 31% this week.", src).unsupported == ["31%"]
    assert check("Sales were +23% vs forecast.", src).unsupported == ["+23%"]


def test_factcheck_ignores_dates_and_citations():
    nums = [raw for raw, *_ in extract_numbers("Since 2015-07-27 [f7, f12] and 05 Apr 2015.")]
    assert nums == []


# ---------- SQL sandbox ----------
@needs_outputs
@pytest.mark.parametrize("q", ["drop table episodes", "select 1; select 2",
                               "select * from read_csv('/etc/hosts')",
                               "select * from '/etc/hosts'", "copy episodes to 'x.csv'"])
def test_run_sql_blocks_unsafe_queries(q):
    from sic.agent.tools import SQLError, run_sql
    with pytest.raises(SQLError):
        run_sql(q)


@needs_outputs
def test_sandbox_blocks_file_access_even_without_filter():
    import duckdb

    from sic.agent.tools import sandbox
    with pytest.raises(duckdb.Error):
        sandbox().execute("select * from read_csv_auto('/etc/hosts')").fetchall()


# ---------- agent loop ----------
@needs_outputs
def test_agent_calls_tool_then_answers_and_factchecks():
    from sic.agent.chat import ask
    llm = FakeLLM([
        {"tool_calls": [{"name": "run_sql", "arguments": {
            "query": "select count(*) n from dim_store where store_type = 'b'"}}]},
        "There are 17 type-b stores.",
    ])
    r = ask("How many type b stores?", llm=llm)
    assert r["tool_calls"][0]["tool"] == "run_sql" and r["tool_calls"][0]["ok"]
    assert r["answer"] == "There are 17 type-b stores."
    assert r["numeric_accuracy"] == 1.0
    tool_msg = next(m for m in llm.requests[1]["messages"] if m.get("role") == "tool")
    assert json.loads(tool_msg["content"])["rows"] == [[17]]


@needs_outputs
def test_agent_flags_hallucinated_number_and_recovers_from_tool_error():
    from sic.agent.chat import ask
    llm = FakeLLM([
        {"tool_calls": [{"name": "run_sql", "arguments": {"query": "drop table episodes"}}]},
        {"tool_calls": [{"name": "run_sql", "arguments": {
            "query": "select count(*) from dim_store"}}]},
        "There are 1,115 stores, growing 40% a year.",
    ])
    r = ask("How many stores?", llm=llm)
    assert [c["ok"] for c in r["tool_calls"]] == [False, True]
    assert r["unsupported_numbers"] == ["40%"]


# ---------- insight cards ----------
@needs_outputs
def test_card_with_wrong_number_is_revised_then_published():
    from sic.insights import cards
    from sic.insights import facts as F
    fp = F.promo2(F.Store())
    eff = fp.facts[0]["display"]
    bad = {"title": "Promo2 lifts sales", "headline": "Promo2 raises sales by +9.0% [f1].",
           "bullets": [], "action": "Run a randomized rollout.", "confidence": "low"}
    good = {**bad, "headline": f"Promo2 may raise sales by about {eff} [f1], but this is "
                               "not proven."}
    llm = FakeLLM([json.dumps(bad), json.dumps({"pass": True, "issues": []}),
                   json.dumps(good), json.dumps({"pass": True, "issues": []})])
    out = cards.write_card(llm, fp)
    assert out["history"][0]["unsupported_numbers"] == ["+9.0%"]
    assert out["qa"]["published"] and out["qa"]["attempts"] == 2
    revise_prompt = llm.requests[2]["messages"][-1]["content"]
    assert "+9.0%" in revise_prompt


@needs_outputs
def test_card_failing_critic_twice_is_held():
    from sic.insights import cards
    from sic.insights import facts as F
    fp = F.promo2(F.Store())
    card = json.dumps({"title": "Promo2 is proven", "headline": "Promo2 definitively works.",
                       "bullets": [], "action": "Enroll every store.", "confidence": "high"})
    fail = json.dumps({"pass": False, "issues": ["overstates certainty"]})
    out = cards.write_card(FakeLLM([card, fail, card, fail]), fp)
    issues = out["qa"]["open_issues"]
    assert not out["qa"]["published"] and "overstates certainty" in issues
    assert any(i.startswith("Overclaiming") for i in issues)  # deterministic hedge rule too


def test_factcheck_ignores_month_year_dates_and_handles_unicode_minus():
    assert check("Earlier episodes (Jan 2015, Dec 2014) were similar.", []).numbers == []
    fc = check("Gap was ‑22 % this week.", [-0.226])  # non-breaking hyphen + nnbsp
    assert fc.unsupported == ["‑22 %"]  # still flagged: wrong rounding of 22.6


@needs_outputs
def test_hedge_rule_flags_unhedged_promo2_claim():
    from sic.insights import cards
    from sic.insights import facts as F
    fp = F.promo2(F.Store())
    bare = {"title": "Promo2 works", "headline": "Promo2 lifts sales.", "bullets": [], "action": ""}
    hedged = {**bare, "headline": "Promo2 may lift sales, but the effect is not proven."}
    assert cards.hedge_check(fp, bare) and not cards.hedge_check(fp, hedged)


@needs_outputs
def test_style_rules_catch_observed_wording_errors():
    from sic.insights import cards
    from sic.insights import facts as F
    s = F.Store()
    ep = F.episode(s, 921, "investigate_underperformance")
    bad = {"headline": "Store 711 missed forecast by -22%.", "bullets": [], "action": ""}
    cite = ep.context["playbook_citation"]
    ok = {"headline": "Store 711 is 22% below forecast.", "bullets": ["Gap is -22%."],
          "action": f"Check local disruptions ({cite})."}
    assert cards.style_check(ep, bad) and not cards.style_check(ep, ok)
    pilot = F.promo_pilot(s)
    assert any("at stake" in i for i in cards.style_check(
        pilot, {"headline": "€405.0k is at risk.", "bullets": [], "action": ""}))
    chain = F.chain_week(s)
    assert cards.style_check(chain, {"headline": "Sales rose 19.9% versus prior week.",
                                     "bullets": [], "action": ""})


@needs_outputs
def test_style_rules_do_not_flag_correct_cards():
    from sic.insights import cards
    from sic.insights import facts as F
    ep = F.episode(F.Store(), 921, "investigate_underperformance")
    ok = {"headline": "Store 921 was 23% below forecast as traffic fell 21% [f3, f11]",
          "bullets": ["Gap vs forecast is -23% [f3]", "22% below target, equal to -€13.3k"],
          "action": f"Check local disruptions ({ep.context['playbook_citation']})."}
    assert cards.style_check(ep, ok) == []
    assert check("Episode ran 2015‑07‑27 to 2015‑08‑02.", []).numbers == []


def _card(published: bool) -> dict:
    return {"subject": "Store 1", "kind": "investigate_underperformance",
            "card": {"title": "t", "headline": "h", "bullets": [], "action": "a",
                     "confidence": "low"},
            "qa": {"published": published, "numeric_accuracy": 1.0, "numbers_checked": 0,
                   "attempts": 1, "open_issues": []}}


def test_degraded_run_never_overwrites_good_snapshot(tmp_path, monkeypatch, capsys):
    from sic.insights import demo
    monkeypatch.setattr(demo, "SNAPSHOT", tmp_path / "snap.json")
    monkeypatch.setattr(demo, "SNAPSHOT_MD", tmp_path / "snap.md")
    assert demo.save_snapshot([_card(True)] * 9 + [_card(False)], "groq", "m")  # 90%
    good = (tmp_path / "snap.json").read_text()
    assert not demo.save_snapshot([_card(True)] * 3 + [_card(False)] * 7, "gemini", "m")
    assert (tmp_path / "snap.json").read_text() == good
    assert demo.show_cached("rate limited") == 0
    out = capsys.readouterr().out
    assert "Live LLM unavailable (rate limited)" in out and "Showing saved demo cards" in out


def test_factcheck_requires_exact_ids_and_counts():
    assert check("Store 1067 is down.", [1066.0]).unsupported == ["1067"]
    assert check("Store 1066 is down; 111 stores in the group.", [1066.0, 111.0]).passed
    assert check("Gap of €13.3k.", [-13326.12]).passed  # money still allows rounding


def test_factcheck_normalizes_unicode_spaces_and_ranges():
    assert check("CI 70.1 – 77.9%", [0.701, 0.779]).passed
    assert check("about 1.4 % over forecast", [0.0142]).passed


def test_factcheck_skips_code_year_month_and_confidence_labels():
    assert check("Filtered with `direction = -1`; no rows for 2016-12.", []).numbers == []
    assert check("Lift +73.97% (95% CI +70.09% to +77.94%).", [0.7397, 0.7009, 0.7794]).passed
    assert check("Sales fell 95% this week.", [0.2]).unsupported == ["95%"]  # still a claim


@needs_outputs
def test_playbook_governance_and_matching():
    from sic.playbooks import library as L
    allowed = {(b["id"], b["version"]) for b in L.eligible()}
    assert ("PB-COMP-NEAR", 1) not in allowed       # superseded
    assert ("PB-HOLIDAY", 0) not in allowed         # approved but past review date
    assert ("PB-PRICE-WAR", 0) not in allowed       # draft
    assert L.match("competitor_opened", "traffic", -1, 100)[0]["citation"] == "PB-COMP-NEAR v2"
    assert L.match("competitor_opened", "traffic", -1, 2500)[0]["citation"] == "PB-COMP-FAR v1"
    assert L.search("rival opened 200 m from the store")[0]["citation"] == "PB-COMP-NEAR v2"
    assert "%" in L.match("competitor_opened", None, -1, 100)[0]["expected_impact"]


@needs_outputs
def test_get_playbook_accepts_null_optionals():
    from sic.agent.chat import _call_tool
    res, ok = _call_tool("get_playbook", json.dumps(
        {"situation": "competitor opened 300 m away", "store_id": None, "driver": None,
         "competitor_distance_m": None}))
    assert ok and res["playbooks"][0]["citation"] == "PB-COMP-NEAR v2"
    res, ok = _call_tool("get_recommendations", json.dumps({"limit": None, "type": None}))
    assert ok and res["recommendations"]


def test_redact_strips_account_ids_and_keys():
    from sic.llm.client import redact
    t = redact("429 for org_01abcDEF in tier on_demand; key gsk_abcdefghijklmnop")
    assert "org_01abcDEF" not in t and "gsk_abcdefghijklmnop" not in t
