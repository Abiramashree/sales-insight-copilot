"""M4: weekly insight cards — plan -> write -> critique -> (revise).

plan      deterministic: which subjects deserve a card (ranked by the pipeline)
write     LLM (JSON mode) turns ONE fact pack into a card, citing fact ids
critique  (a) numeric fact-check: every number must match a fact   [deterministic]
          (b) LLM reviewer: unsupported claims, causal overreach, tone [LLM, JSON]
revise    if either fails, rewrite once with the specific issues

Writes data/outputs/insight_cards.json and reports/insight_cards.md
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from sic import config
from sic.insights import facts as F
from sic.insights.factcheck import check
from sic.llm.client import redact

WRITER_SYSTEM = """You write weekly sales insight cards for regional sales managers.

Rules:
- Use ONLY the facts provided. Never compute, estimate or round new numbers.
- Copy numbers exactly as they appear in each fact's "display" field.
- After every sentence that contains a number, cite the fact id(s) like [f3] or [f3, f5].
- Plain business language, no jargon (say "customer visits", not "traffic log gap").
- Match certainty to the evidence: 1-2 week signals are "early signals"; anything marked
  as uncertain or non-definitive in context must be described that way, in the headline.
- Keep each number's meaning exactly as its label says (a daily figure is not "weekly";
  "at stake" is not "at risk"; one store type's figure is not a group's figure).
- Do not connect facts causally unless a fact states the link (e.g. a store's general
  promo lift does not explain one week's gap).
- Write direction in words OR as a sign, never both ("23% below forecast", not "missed by -23%").
- One concrete action the manager can take this week that follows from the facts. If an
  approved playbook is in the facts, the action MUST be one of its plays and end with the
  playbook citation in parentheses, e.g. "(PB-COMP-NEAR v2)". Never invent other actions.

Return JSON only:
{"title": "<= 8 words",
 "headline": "one sentence, the single most important point",
 "bullets": ["2-3 short supporting points"],
 "action": "one sentence",
 "cited_facts": ["f1", ...]}"""

CRITIC_SYSTEM = """You are a strict reviewer of a sales insight card. Check it against the facts
AND the "context" block (context notes are binding). Go through every check:
1. Unsupported claim: any statement, cause or comparison no fact states.
2. Causal leap: two facts linked as cause/effect without a fact stating the link.
3. Mislabelled number: a number described differently from its fact label (daily vs weekly,
   one segment vs a group, "at stake" vs "at risk", episode vs week).
4. Overclaiming: certainty stronger than the facts/context allow. If context says an estimate
   is uncertain or not proven, the card MUST say so; omitting the caveat is a failure.
5. Direction error: up vs down, or a confusing double negative ("missed by -23%").
6. Action that does not follow from the facts. (Proposing it "this week" is expected; do not
   flag timing. Do not judge the confidence level; it is set separately from the evidence.)
Return JSON only: {"pass": true|false, "issues": ["<check #>: specific issue", ...]}
pass is true only if no check fails."""

HEDGE_WORDS = ("suggest", "may ", "might", "uncertain", "not proven", "unproven", "likely",
               "preliminary", "tentative", "early signal", "not definitive", "inconclusive")


def plan(s: F.Store) -> list[F.FactPack]:
    cfg = config.load()["insights"]
    packs = [F.chain_week(s)]
    seg = F.segment_anomaly(s)
    if seg:
        packs.append(seg)
    packs += [F.promo_pilot(s), F.promo2(s)]
    under = s.recs[s.recs.type == "investigate_underperformance"].head(
        cfg["n_underperformance_cards"])
    over = s.recs[s.recs.type == "replicate_outperformance"].head(cfg["n_outperformance_cards"])
    packs += [F.episode(s, int(r.store_id), r.type) for _, r in under.iterrows()]
    packs += [F.episode(s, int(r.store_id), r.type) for _, r in over.iterrows()]
    # competitive response: most recent below-forecast episode caused by a new competitor
    seen = {int(p.facts[0]["value"]) for p in packs if p.facts and p.facts[0]["label"] == "Store ID"}
    comp = s.erca[(s.erca.root_cause == "competitor_opened") & (s.erca.direction < 0)
                  & ~s.erca.store_id.isin(seen)].sort_values(["end", "severity"],
                                                             ascending=False)
    if len(comp):
        packs.append(F.episode(s, int(comp.iloc[0].store_id), "competitor_response"))
    return packs


def card_text(card: dict) -> str:
    return " ".join([card.get("title", ""), card.get("headline", ""),
                     *card.get("bullets", []), card.get("action", "")])


def _parse(content: str | None) -> dict:
    if not content:
        raise ValueError("empty LLM response")
    content = content.strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)  # tolerate code fences
    return json.loads(m.group(0) if m else content)


def write_card(llm, fp: F.FactPack) -> dict:
    max_rev = config.load()["insights"]["max_revisions"]
    messages = [{"role": "system", "content": WRITER_SYSTEM},
                {"role": "user", "content": f"Facts:\n{fp.to_prompt()}"}]
    t0, attempts, history = time.time(), 0, []
    while True:
        attempts += 1
        try:
            raw = llm.chat(messages, json_mode=True, purpose=f"card_write:{fp.kind}").content
        except Exception as e:  # noqa: BLE001 - one failed card must not sink the run
            raw = None
            llm_error = redact(f"LLM error while writing: {type(e).__name__}: {str(e)[:200]}")
        else:
            llm_error = None
        try:
            card = _parse(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            card = {"title": "", "headline": "", "bullets": [], "action": "", "_raw": raw}
        fc = check(card_text(card), fp.values())
        rule_issues = hedge_check(fp, card) + style_check(fp, card)
        if llm_error or not card.get("headline"):
            review = {"pass": False, "issues": [llm_error or "Writer returned no usable card JSON."]}
        else:
            review = critique(llm, fp, card)
        issues = ([f"Number not in facts: {n}" for n in fc.unsupported] + rule_issues
                  + review.get("issues", []))
        history.append({"attempt": attempts, "numeric_accuracy": fc.accuracy,
                        "unsupported_numbers": fc.unsupported, "critic_pass": review.get("pass"),
                        "issues": issues})
        if not issues or attempts > max_rev:
            break
        messages += [{"role": "assistant", "content": raw or ""},
                     {"role": "user", "content": "Revise the card. Fix these issues and keep "
                      "everything else:\n- " + "\n- ".join(issues)}]
    card["confidence"] = confidence(fp)
    return {
        "subject": fp.subject, "kind": fp.kind, "card": card,
        "qa": {"numeric_accuracy": fc.accuracy, "numbers_checked": len(fc.numbers),
               "unsupported_numbers": fc.unsupported, "critic_pass": bool(review.get("pass")),
               "open_issues": issues, "attempts": attempts,
               "published": not issues, "latency_s": round(time.time() - t0, 2)},
        "history": history,
        "facts": fp.facts,
    }


_NEG_WORDS = r"(?:missed|miss|below|down|fell|fall|dropped|drop|declin\w*|short\w*|under\w*|lost)"
# the observed error: "missed forecast by -22%" (direction stated twice)
_DOUBLE_NEG = re.compile(_NEG_WORDS + r"\b[^.;\[]{0,30}?\bby\s*[-−–‑‒]\s?€?\s?\d", re.IGNORECASE)


def confidence(fp: F.FactPack) -> str:
    """Deterministic: from the evidence, never from the writer."""
    if fp.kind == "promo2_program":
        return "low"  # pre-trend test failed
    if fp.kind in ("investigate_underperformance", "replicate_outperformance",
                   "competitor_response"):
        weeks = next((f["value"] for f in fp.facts if f["label"] == "Consecutive anomalous weeks"),
                     1)
        return "high" if weeks >= 3 else "medium" if weeks == 2 else "low"
    return {"segment_anomaly": "high", "chain_weekly": "high"}.get(fp.kind, "medium")


def style_check(fp: F.FactPack, card: dict) -> list[str]:
    """Deterministic wording rules the LLM critic was observed to miss."""
    text = card_text(card)
    issues = []
    m = _DOUBLE_NEG.search(text)
    if m:
        issues.append(f"Double negative: '{m.group(0)}'. Use words OR a sign, e.g. "
                      "'22% below forecast'.")
    labels = " ".join(f["label"].lower() for f in fp.facts)
    if "at stake" in labels and re.search(r"\brisk", text, re.IGNORECASE):
        issues.append("Mislabel: the fact says revenue 'at stake' (exposed to the test), "
                      "not 'at risk' (likely to be lost).")
    cite = fp.context.get("playbook_citation")
    if cite and cite not in card.get("action", ""):
        issues.append(f"The action must come from the approved playbook and cite ({cite}).")
    note = json.dumps(fp.context).lower()
    if "not raw week-over-week" in note and re.search(r"prior week|previous week|last week|"
                                                      r"week-over-week|wow", card.get("headline", ""),
                                                      re.IGNORECASE):
        issues.append("The week may be partial: lead with performance vs forecast, not the "
                      "comparison with the prior week.")
    return issues


def hedge_check(fp: F.FactPack, card: dict) -> list[str]:
    """Deterministic guard: if the facts say an estimate is not proven, the card must hedge."""
    ctx = json.dumps(fp.context).lower()
    if "not be described as proven" not in ctx and "non-definitive" not in ctx:
        return []
    text = card_text(card).lower()
    if any(w in text for w in HEDGE_WORDS):
        return []
    return [("Overclaiming: the facts say this estimate is not proven; the card must say so "
            "(e.g. 'suggestive', 'not proven', 'may').")]


def critique(llm, fp: F.FactPack, card: dict) -> dict:
    try:
        return _critique(llm, fp, card)
    except Exception as e:  # noqa: BLE001 - a critic outage holds the card, never crashes
        return {"pass": False, "issues": [f"Critic unavailable: {type(e).__name__}"]}


def _critique(llm, fp: F.FactPack, card: dict) -> dict:
    raw = llm.chat([{"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": f"Facts:\n{fp.to_prompt()}\n\nCard:\n"
                                                f"{json.dumps(card, indent=1)}"}],
                   json_mode=True, purpose=f"card_critic:{fp.kind}", temperature=0,
                   model=getattr(llm, "cfg", {}).get("critic_model"),
                   reasoning_effort=config.load()["insights"].get("critic_reasoning_effort",
                                                                   "config")).content
    try:
        r = _parse(raw)
        return {"pass": bool(r.get("pass")), "issues": list(r.get("issues") or [])}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"pass": False, "issues": ["critic returned unparseable output"]}


def render_md(cards: list[dict]) -> str:
    strip = lambda t: re.sub(r"\s*\[f\d+(?:\s*,\s*f\d+)*\]", "", t)
    ok = sum(c["qa"]["published"] for c in cards)
    acc = sum(c["qa"]["numeric_accuracy"] for c in cards) / max(len(cards), 1)
    out = [(f"# Weekly insight cards\n\n{ok}/{len(cards)} cards passed QA · mean numeric "
           f"accuracy {acc:.0%}\n")]
    for c in cards:
        k, q = c["card"], c["qa"]
        badge = "✅ published" if q["published"] else "⚠️ held for review"
        out.append(f"## {strip(k.get('title', c['subject']))}\n\n*{c['subject']} · {badge} · "
                   f"confidence {k.get('confidence', '?')} · numbers {q['numbers_checked']} "
                   f"checked, accuracy {q['numeric_accuracy']:.0%} · attempts {q['attempts']}*\n")
        out.append(f"**{strip(k.get('headline', ''))}**\n")
        out += [f"- {strip(b)}" for b in k.get("bullets", [])]
        out.append(f"\n**Action:** {strip(k.get('action', ''))}\n")
        if q["open_issues"]:
            out.append("> QA issues: " + "; ".join(q["open_issues"]) + "\n")
    return "\n".join(out)


def run(llm=None) -> list[dict]:
    from sic.llm.client import get_llm

    llm = llm or get_llm()
    packs = plan(F.Store())
    workers = config.load()["insights"].get("parallel_cards", 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:  # keeps plan order
        cards = list(pool.map(lambda fp: write_card(llm, fp), packs))
    (config.path("outputs") / "insight_cards.json").write_text(
        json.dumps(cards, indent=1, default=str))
    (config.path("reports") / "insight_cards.md").write_text(render_md(cards))
    from sic.insights.demo import save_snapshot
    save_snapshot(cards, getattr(llm, "cfg", {}).get("provider", "?"),
                  getattr(llm, "model", "?"))
    return cards


if __name__ == "__main__":
    cs = run()
    for c in cs:
        print(f"[{'OK ' if c['qa']['published'] else 'HOLD'}] {c['subject']}: "
              f"{c['card'].get('headline', '')}")
