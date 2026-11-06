"""M3: turn detection, RCA and causal estimates into ranked, sized actions.

Every recommendation carries: what to do, why (evidence), € impact per week,
confidence, and the source numbers — so the M4 LLM layer can narrate it and
the eval harness can fact-check the narration against these fields.

Writes data/outputs/recommendations.parquet and reports/m3_recommendations.json
"""

from __future__ import annotations

import json

import pandas as pd

from sic import config


def run() -> dict:
    out, rep = config.path("outputs"), config.path("reports")
    lift = pd.read_parquet(out / "promo_lift_by_store.parquet")
    erca = pd.read_parquet(out / "episode_rca.parquet")
    did = json.loads((rep / "m3_promo2_did.json").read_text())
    recs: list[dict] = []
    last_week = erca.end.max()

    # 1. Ongoing underperformance at the end of the data -> investigate
    live = erca[(erca.end >= last_week - pd.Timedelta(days=13))]
    for _, e in live[live.direction < 0].iterrows():
        recs.append({
            "type": "investigate_underperformance", "store_id": int(e.store_id),
            "action": f"Investigate store {e.store_id}: {e.gap_pct:+.0%} vs forecast for "
                      f"{e.weeks} week(s), {e.tb_driver}-driven.",
            "weekly_eur_impact": float(abs(e.gap) / e.weeks),
            "confidence": "high" if e.weeks >= 3 else "medium",
            "evidence": {"root_cause": e.root_cause, "gap_pct": float(e.gap_pct),
                         "weeks": int(e.weeks), "traffic_share": float(e.tb_traffic_share),
                         "events": json.loads(e.evidence)},
        })
    # 2. Ongoing outperformance -> learn and replicate
    for _, e in live[live.direction > 0].iterrows():
        recs.append({
            "type": "replicate_outperformance", "store_id": int(e.store_id),
            "action": f"Study store {e.store_id}: {e.gap_pct:+.0%} vs forecast for "
                      f"{e.weeks} week(s); identify what is working and replicate.",
            "weekly_eur_impact": float(abs(e.gap) / e.weeks) * 0.25,  # assume 25% transferable
            "confidence": "low",
            "evidence": {"root_cause": e.root_cause, "gap_pct": float(e.gap_pct),
                         "weeks": int(e.weeks), "events": json.loads(e.evidence)},
        })

    # 3. Promo targeting pilot: lowest-lift decile
    low = lift[lift.lift_hi < lift.lift.quantile(0.25)].nsmallest(
        max(1, len(lift) // 10), "lift")
    recs.append({
        "type": "promo_targeting_pilot", "store_id": None,
        "action": f"A/B-test promo mechanics in the {len(low)} lowest-lift stores "
                  f"(median lift {low.lift.median():+.0%} vs chain {lift.lift.median():+.0%}); "
                  "randomize half to an alternative mechanic for 2 promo cycles.",
        "weekly_eur_impact": None,  # gain is unknown until the test runs
        "confidence": "medium",
        "evidence": {"stores": low.store_id.tolist()[:20],
                     # promo-driven revenue exposed in these stores (avg week; promos
                     # run every other week) — what the test puts at stake, not a gain
                     "weekly_promo_revenue_at_stake": float(
                         low.incremental_eur_per_promo_week.sum() / 2),
                     "share_type_b_or_extra": float(((low.store_type == "b")
                                                     | (low.assortment == "extra")).mean()),
                     "caveat": "Lift is vs. non-promo weeks and may include pull-forward; "
                               "no margin data, so € is revenue, not profit."},
    })
    # 4. Protect execution in top responders
    top = lift.nlargest(25, "incremental_eur_per_promo_week")
    recs.append({
        "type": "protect_promo_execution", "store_id": None,
        "action": f"Prioritize stock & staffing on promo weeks in the top {len(top)} "
                  f"responders (median lift {top.lift.median():+.0%}).",
        "weekly_eur_impact": float(top.incremental_eur_per_promo_week.sum() / 2 * 0.05),
        "confidence": "low",
        "evidence": {"stores": top.store_id.tolist(),
                     "assumption": "5% of incremental promo revenue lost to stock-outs today."},
    })
    # 5. Promo2 program decision
    eff, pre = did["effect"], did["pretrend_test"]
    recs.append({
        "type": "promo2_program", "store_id": None,
        "action": "Do not scale Promo2 on observational evidence alone; run a randomized "
                  "staggered rollout to settle the effect.",
        "weekly_eur_impact": None,
        "confidence": "low",
        "evidence": {"did_effect_pct": eff["effect_pct"], "ci_pct": eff["ci_pct"],
                     "pretrend_p": pre["p_value"],
                     "placebo_effect_pct": did["placebo"]["effect_pct"],
                     "robustness": {k: v["effect_pct"] for k, v in did["robustness"].items()},
                     "reason": "Positive in every spec, but enrolled stores were sliding "
                               "before enrollment (pre-trend p<0.01), so part may be rebound."},
    })

    r = pd.DataFrame(recs)
    r["rank"] = r.weekly_eur_impact.rank(ascending=False, method="first", na_option="bottom")
    r = r.sort_values("rank").reset_index(drop=True)
    r_out = r.assign(evidence=r.evidence.map(json.dumps))
    r_out.to_parquet(out / "recommendations.parquet", index=False)

    summary = {
        "as_of_week": str(last_week.date()),
        "n_recommendations": len(r),
        "by_type": r.type.value_counts().to_dict(),
        "top10": json.loads(r.head(10).drop(columns="evidence").to_json(orient="records")),
        "portfolio": [x for x in recs if x["store_id"] is None],
    }
    (rep / "m3_recommendations.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps({k: s[k] for k in ("as_of_week", "n_recommendations", "by_type")}, indent=2))
    for x in s["top10"]:
        imp = x["weekly_eur_impact"]
        print(f"- [{x['type']}] {x['action']}  (€{imp:,.0f}/wk, {x['confidence']})"
              if imp else f"- [{x['type']}] {x['action']}")
