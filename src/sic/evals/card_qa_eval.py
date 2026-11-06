"""Card QA gate eval: inject known defects into correct cards and measure what gets caught.

Clean cards are built from real fact packs by templates (so they are correct by
construction). Each defect injector produces a copy with exactly one known problem.

  deterministic gate = numeric fact-check + hedge rule + style rules   (no LLM)
  full gate          = deterministic OR LLM critic                     (--llm)

Reports detection rate per defect type and the false-positive rate on clean cards.
Some defects (causal leaps, invented causes) are expected to need the LLM critic.
"""

from __future__ import annotations

import copy
import json
import re
import sys

from sic import config
from sic.insights import cards as C
from sic.insights import facts as F
from sic.insights.factcheck import check


def _get(fp: F.FactPack, label: str, default=None):
    f = next((f for f in fp.facts if label.lower() in f["label"].lower()), None)
    return (f["display"], f["value"]) if f else (default, None)


def _unsigned(d: str) -> str:
    return d.lstrip("+-−")


# ------------------------------------------------------------------ clean cards
def clean_card(fp: F.FactPack) -> dict | None:
    k = fp.kind
    if k in ("investigate_underperformance", "replicate_outperformance", "competitor_response"):
        sid = _get(fp, "Store ID")[0]
        gap, gv = _get(fp, "Gap vs forecast")
        weeks = _get(fp, "Consecutive anomalous weeks")[1] or 1
        traffic = _get(fp, "Customer traffic vs forecast")[0]
        word = "below" if gv < 0 else "above"
        signal = "an early signal" if weeks <= 2 else f"for {weeks} weeks running"
        return {"title": f"Store {sid} {word} forecast",
                "headline": f"Store {sid} is {_unsigned(gap)} {word} forecast, {signal}.",
                "bullets": [f"Customer visits were {traffic} versus forecast.",
                            f"The root cause is {_get(fp, 'Root-cause classification')[0]}."],
                "action": (f"{_get(fp, 'Playbook play 1')[0].split(' (')[0]} "
                           f"({fp.context['playbook_citation']})."
                           if fp.context.get("playbook_citation")
                           else "Review local traffic drivers with the store manager this week.")}
    if k == "promo2_program":
        e = _get(fp, "Estimated effect")[0]
        lo, hi = _get(fp, "95% CI low")[0], _get(fp, "95% CI high")[0]
        return {"title": "Promo2 effect is uncertain",
                "headline": f"Promo2 may lift sales by about {e}, but the effect is not proven.",
                "bullets": [f"The estimate ranges from {lo} to {hi}.",
                            "Enrolled stores were declining before enrollment."],
                "action": "Run a randomized rollout before expanding Promo2."}
    if k == "promo_targeting_pilot":
        n = _get(fp, "Stores in lowest-lift group")[0]
        stake = _get(fp, "revenue at stake")[0]
        low, chain = _get(fp, "Median lift in that group")[0], _get(fp, "Chain median store lift")[0]
        return {"title": "Test promo mechanics in low-lift stores",
                "headline": f"{n} low-lift stores have {stake} of weekly promo revenue at stake.",
                "bullets": [f"Their median promo lift is {low} versus {chain} chain-wide."],
                "action": "Randomize half of them to an alternative mechanic for 2 promo cycles."}
    if k == "chain_weekly":
        sales, vs = _get(fp, "Chain sales this week")[0], _get(fp, "Chain sales vs forecast")[0]
        yoy = _get(fp, "Same-store sales vs same days last year")[0]
        b = [f"Same-store sales were {yoy} versus the same days last year."] if yoy else []
        return {"title": "Chain week vs forecast",
                "headline": f"Chain sales were {sales}, {vs} versus forecast.", "bullets": b,
                "action": "Review the stores flagged below forecast this week."}
    if k == "segment_anomaly":
        g = _get(fp, "Segment sales vs forecast")[0]
        share = _get(fp, "Share of stores moving the same way")[0]
        seg = fp.subject.split(",")[0]
        return {"title": f"{seg} holiday-week swing",
                "headline": f"{seg} sales were {g} versus forecast.",
                "bullets": [f"{share} of stores moved the same way, pointing to holiday timing."],
                "action": "Plan stock for the same holiday window next year."}
    return None


# ------------------------------------------------------------------ defect injectors
PCT = re.compile(r"(\d+(?:\.\d+)?)%")


def wrong_number(fp, c):
    m = PCT.search(c["headline"])
    if not m:
        return None
    v = float(m.group(1)) + 7
    c["headline"] = c["headline"][:m.start(1)] + (f"{v:.1f}" if "." in m.group(1) else f"{v:.0f}") \
        + c["headline"][m.end(1):]
    return c


def sign_flip(fp, c):
    for label in ("Gap vs forecast", "Segment sales vs forecast", "Chain sales vs forecast"):
        d, v = _get(fp, label)
        if d and isinstance(v, float) and abs(v) > 0.005:
            c["bullets"].append(f"The gap versus forecast was {'+' if v < 0 else '-'}"
                                f"{_unsigned(d)}.")
            return c
    return None


def store_id_typo(fp, c):
    sid = _get(fp, "Store ID")[0]
    if not sid:
        return None
    bad = sid[:-2] + sid[-1] + sid[-2] if len(sid) >= 2 and sid[-1] != sid[-2] else str(int(sid) + 1)
    c["headline"] = c["headline"].replace(f"Store {sid}", f"Store {bad}")
    return c


def double_negative(fp, c):
    d, _v = _get(fp, "Gap vs forecast")
    if fp.kind != "investigate_underperformance" or not d:
        return None
    c["headline"] = f"Store {_get(fp, 'Store ID')[0]} missed forecast by {d}."
    return c


def at_risk(fp, c):
    if fp.kind != "promo_targeting_pilot":
        return None
    c["headline"] = c["headline"].replace("at stake", "at risk")
    return c


def missing_hedge(fp, c):
    if fp.kind != "promo2_program":
        return None
    e = _get(fp, "Estimated effect")[0]
    c["headline"] = f"Promo2 lifts sales by {e}."
    c["title"] = "Promo2 works"
    c["bullets"] = [b for b in c["bullets"] if "declining" not in b]
    c["action"] = "Expand Promo2 to more stores."
    return c


def wow_lead(fp, c):
    d = _get(fp, "Chain sales vs prior week")[0]
    if fp.kind != "chain_weekly" or not d:
        return None
    c["headline"] = f"Chain sales rose {_unsigned(d)} versus the prior week."
    return c


def uncited_action(fp, c):  # action not from the approved playbook
    if not fp.context.get("playbook_citation"):
        return None
    c["action"] = "Launch a 10% store-wide discount this week."
    return c


def causal_leap(fp, c):  # expected to need the LLM critic
    lift = _get(fp, "Store's causal promo lift")[0]
    if not lift or fp.kind != "investigate_underperformance":
        return None
    c["bullets"].append(f"The shortfall is caused by the store's weak promo lift of {lift}.")
    return c


def invented_cause(fp, c):  # expected to need the LLM critic
    if fp.kind not in ("investigate_underperformance", "replicate_outperformance"):
        return None
    if any("competitor" in (f["label"] + str(f["display"])).lower() for f in fp.facts):
        return None
    c["bullets"].append("A new competitor opening nearby explains the change.")
    return c


def overclaim(fp, c):  # expected to need the LLM critic
    weeks = _get(fp, "Consecutive anomalous weeks")[1]
    if fp.kind != "investigate_underperformance" or not weeks or weeks > 2:
        return None
    c["headline"] = c["headline"].replace("an early signal", "a confirmed long-term decline")
    return c


LLM_ONLY = {"causal_leap", "invented_cause", "overclaim"}  # beyond rule-based checks
INJECTORS = {f.__name__: f for f in (wrong_number, sign_flip, store_id_typo, double_negative,
                                      at_risk, missing_hedge, wow_lead, uncited_action, causal_leap,
                                      invented_cause, overclaim)}


# ------------------------------------------------------------------ gate
def deterministic_issues(fp: F.FactPack, card: dict) -> list[str]:
    fc = check(C.card_text(card), fp.values())
    return ([f"number: {n}" for n in fc.unsupported] + C.hedge_check(fp, card)
            + C.style_check(fp, card))


def build_cases() -> list[dict]:
    s = F.Store()
    packs = C.plan(s)
    cases = []
    for fp in packs:
        base = clean_card(fp)
        if not base:
            continue
        cases.append({"fp": fp, "card": base, "defect": "clean"})
        for name, inj in INJECTORS.items():
            bad = inj(fp, copy.deepcopy(base))
            if bad is not None and bad != base:
                cases.append({"fp": fp, "card": bad, "defect": name})
    return cases


def run(use_llm: bool = False) -> dict:
    cases = build_cases()
    llm = None
    if use_llm:
        from sic.llm.client import get_llm
        llm = get_llm()
    rows = []
    for c in cases:
        det = deterministic_issues(c["fp"], c["card"])
        crit = C.critique(llm, c["fp"], c["card"]) if llm else None
        crit_err = bool(crit) and any(i.startswith(("Critic unavailable", "critic returned"))
                                      for i in crit["issues"])
        rows.append({"defect": c["defect"], "subject": c["fp"].subject,
                     "det_flag": bool(det), "det_issues": det,
                     "critic_error": crit_err,
                     "critic_flag": None if (crit is None or crit_err) else not crit["pass"],
                     "critic_issues": crit["issues"] if crit else None})
    by: dict = {}
    for r in rows:
        b = by.setdefault(r["defect"], {"n": 0, "det": 0, "critic": 0, "either": 0})
        b["n"] += 1
        b["det"] += r["det_flag"]
        if use_llm:
            b.setdefault("scored", 0)
            if r["critic_flag"] is not None:  # critic errors are excluded, reported apart
                b["scored"] += 1
                b["critic"] += r["critic_flag"]
                b["either"] += r["det_flag"] or r["critic_flag"]
    summary = {d: {"n": v["n"], "deterministic": v["det"] / v["n"],
                   **({"llm_critic": v["critic"] / max(v["scored"], 1),
                       "full_gate": v["either"] / max(v["scored"], 1)} if use_llm else {})}
               for d, v in by.items()}
    defects = [r for r in rows if r["defect"] != "clean"]
    clean = [r for r in rows if r["defect"] == "clean"]
    result = {
        "cases": len(rows), "clean_cards": len(clean), "defective_cards": len(defects),
        "by_defect": summary,
        "deterministic_recall": sum(r["det_flag"] for r in defects) / len(defects),
        "deterministic_false_positive_rate": sum(r["det_flag"] for r in clean) / len(clean),
        "clean_false_positives": [r for r in clean if r["det_flag"]],
    }
    if use_llm:
        sd = [r for r in defects if r["critic_flag"] is not None]
        sc = [r for r in clean if r["critic_flag"] is not None]
        llm_only = [r for r in sd if r["defect"] in LLM_ONLY]
        result |= {
            "critic_errors": sum(r["critic_error"] for r in rows),
            "full_gate_recall": sum(r["det_flag"] or r["critic_flag"] for r in sd) / len(sd),
            "critic_recall_on_llm_only_defects": sum(r["critic_flag"] for r in llm_only)
            / max(len(llm_only), 1),
            "critic_false_positive_rate": sum(r["critic_flag"] for r in sc) / max(len(sc), 1),
            "critic_false_positives": [r for r in sc if r["critic_flag"]],
        }
    name = "m5_card_qa_eval_llm.json" if use_llm else "m5_card_qa_eval.json"
    (config.path("reports") / name).write_text(json.dumps(result, indent=1, default=str))
    return result


if __name__ == "__main__":
    r = run(use_llm="--llm" in sys.argv)
    print(json.dumps({k: v for k, v in r.items() if "positives" not in k}, indent=1))
    for fp in r.get("clean_false_positives", []):
        print("CLEAN FP:", fp["subject"], fp["det_issues"])
