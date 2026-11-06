"""What-if scenario engine: causal estimates x store baselines, with Monte Carlo uncertainty.

The LLM never computes scenario numbers. It translates a question into a Scenario;
this engine applies effects that were *measured* elsewhere in the pipeline:

  extra_promo_week / remove_promo_week   per-store causal promo lift (within-store
                                         regression, src/sic/causal/promo_lift.py)
  competitor_opens                       stacked DiD of 120 competitor openings, split
                                         within / beyond 1 km (competitor_impact.py)
  promo2_enroll                          stacked DiD of Promo2 enrollment (promo2_did.py)
                                         -- flagged low confidence (pre-trends fail)

Uncertainty: effects are drawn from their sampling distributions (log scale,
N(beta, se)); totals are summed per draw, so the reported range reflects estimate
uncertainty across all selected stores. Baselines: recent out-of-sample forecasts
(competitor / Promo2) or 2015 non-promo weekday sales (promo).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from sic import config

SCENARIOS = {
    "extra_promo_week": "Add promo week(s) (Mon-Fri promo) in the selected stores",
    "remove_promo_week": "Cancel promo week(s) in the selected stores",
    "competitor_opens": "A new competitor opens near the selected stores",
    "promo2_enroll": "Enroll the selected stores in the Promo2 program",
}
N_DRAWS = 5000


class ScenarioError(ValueError):
    pass


@dataclass
class Scenario:
    scenario: str
    store_ids: list[int] | None = None
    store_type: str | None = None
    lowest_lift_n: int | None = None
    weakest_n: int | None = None
    all_stores: bool = False
    weeks: int | None = None
    competitor_distance_m: float | None = None
    seed: int = 7
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ inputs
@lru_cache
def _inputs() -> dict:
    out, rep = config.path("outputs"), config.path("reports")
    lift = pd.read_parquet(out / "promo_lift_by_store.parquet").set_index("store_id")
    w = pd.read_parquet(out / "store_weeks.parquet")
    full = w[w.open_days >= 5]
    recent = full[full.week_start >= full.week_start.max() - pd.Timedelta(weeks=7)]
    base = recent.groupby("store_id").forecast.mean().rename("weekly_baseline")
    gap = recent.groupby("store_id").apply(
        lambda g: g.sales.sum() / g.forecast.sum() - 1, include_groups=False).rename("recent_gap")
    con = config.connect(read_only=True)
    dim = con.execute("select store_id, store_type, assortment, promo2_enrolled, "
                      "competition_distance_m from marts.dim_store").df().set_index("store_id")
    con.close()
    comp = json.loads((rep / "m3_competitor_impact.json").read_text())
    p2 = json.loads((rep / "m3_promo2_did.json").read_text())
    return {"lift": lift, "base": base, "gap": gap, "dim": dim, "comp": comp, "p2": p2,
            "baseline_weeks": f"{recent.week_start.min().date()} to {recent.week_start.max().date()}"}


def _log_se_from_ci(ci_pct: list[float]) -> float:
    lo, hi = np.log1p(ci_pct[0]), np.log1p(ci_pct[1])
    return float((hi - lo) / (2 * 1.959964))


# ------------------------------------------------------------------ store selection
def select_stores(s: Scenario) -> tuple[list[int], str]:
    d = _inputs()
    universe = d["dim"].index.intersection(d["base"].index)
    if s.store_ids:
        bad = [i for i in s.store_ids if i not in universe]
        if bad:
            raise ScenarioError(f"Unknown or inactive store ids: {bad[:10]} (valid 1-1115).")
        return list(dict.fromkeys(int(i) for i in s.store_ids)), f"stores {s.store_ids[:10]}"
    if s.lowest_lift_n:
        ids = d["lift"].loc[d["lift"].index.intersection(universe)].nsmallest(
            int(s.lowest_lift_n), "lift").index.tolist()
        return ids, f"the {len(ids)} stores with the lowest causal promo lift"
    if s.weakest_n:
        ids = d["gap"].loc[universe].nsmallest(int(s.weakest_n)).index.tolist()
        return ids, f"the {len(ids)} stores furthest below forecast over the last 8 weeks"
    if s.store_type:
        ids = [i for i in universe if d["dim"].loc[i, "store_type"] == s.store_type.lower()]
        if not ids:
            raise ScenarioError(f"No stores of type {s.store_type!r} (types: a, b, c, d).")
        return ids, f"all {len(ids)} type-{s.store_type.lower()} stores"
    if s.all_stores:
        return list(universe), f"all {len(universe)} stores"
    raise ScenarioError("Choose stores: store_ids, store_type, lowest_lift_n, weakest_n "
                        "or all_stores.")


# ------------------------------------------------------------------ simulation
def _summary(draws: np.ndarray) -> dict:
    return {"mean": float(draws.mean()), "p5": float(np.percentile(draws, 5)),
            "p95": float(np.percentile(draws, 95))}


def simulate(s: Scenario) -> dict:
    if s.scenario not in SCENARIOS:
        raise ScenarioError(f"Unsupported scenario {s.scenario!r}. Supported: "
                            + "; ".join(f"{k} ({v})" for k, v in SCENARIOS.items()))
    d = _inputs()
    ids, selection = select_stores(s)
    rng = np.random.default_rng(s.seed)
    assumptions, caveats = [], []

    if s.scenario in ("extra_promo_week", "remove_promo_week"):
        weeks = int(s.weeks or 1)
        L = d["lift"].loc[d["lift"].index.intersection(ids)]
        sign = 1 if s.scenario == "extra_promo_week" else -1
        base_wk = (L.avg_nonpromo_weekday_sales * 5).values          # 5 promo days / week
        beta = rng.normal(L.beta.values, L.se.values, size=(N_DRAWS, len(L)))
        per_store = sign * weeks * base_wk * np.expm1(beta)           # draws x stores
        baseline_total = float(base_wk.sum() * weeks)
        effect_desc = (f"store-specific causal promo lift (median {np.expm1(L.beta).median():+.1%}, "
                       f"range {np.expm1(L.beta).min():+.1%} to {np.expm1(L.beta).max():+.1%})")
        method = "Within-store promo regression (src/sic/causal/promo_lift.py)"
        confidence = "medium"
        assumptions += ["The added/cancelled weeks would otherwise be non-promo/promo weeks.",
                        "Baseline = 2015 average non-promo weekday sales x 5 promo days."]
        caveats += [("Lift is measured against non-promo weeks and may include pull-forward "
                    "(customers shifting purchases), so the net gain can be smaller."),
                    "Revenue only: no margin or promo cost data."]
        store_ids = L.index.tolist()
    elif s.scenario == "competitor_opens":
        if s.competitor_distance_m is None:
            raise ScenarioError("competitor_opens needs competitor_distance_m (metres).")
        weeks = int(s.weeks or 26)
        near = s.competitor_distance_m < 1000
        band = d["comp"]["by_distance"]["within_1000m" if near else "beyond_1000m"]
        mu, se = np.log1p(band["effect_pct"]), _log_se_from_ci(band["ci_pct"])
        base_wk = d["base"].loc[ids].values
        eff = rng.normal(mu, se, size=(N_DRAWS, 1))                   # one shared effect
        per_store = weeks * base_wk * np.expm1(eff)
        baseline_total = float(base_wk.sum() * weeks)
        effect_desc = (f"{band['effect_pct']:+.1%} per week (95% CI {band['ci_pct'][0]:+.1%} "
                       f"to {band['ci_pct'][1]:+.1%}), {'within' if near else 'beyond'} 1 km")
        method = (f"Stacked difference-in-differences of {d['comp']['treated_stores']} competitor "
                  "openings (src/sic/causal/competitor_impact.py)")
        confidence = "high" if d["comp"].get("credible") else "medium"
        assumptions += [(f"Competitor {s.competitor_distance_m:.0f} m away; effect held for "
                        f"{weeks} weeks (the measured drop showed no recovery over 6 months)."),
                        f"Baseline = average weekly forecast, {d['baseline_weeks']}."]
        caveats += ["Average effect across past openings; a specific competitor can differ.",
                    "Opening dates in the source data are month-level."]
        store_ids = ids
    else:  # promo2_enroll
        weeks = int(s.weeks or 26)
        already = [i for i in ids if bool(d["dim"].loc[i, "promo2_enrolled"])]
        ids = [i for i in ids if i not in already]
        if not ids:
            raise ScenarioError("All selected stores are already enrolled in Promo2.")
        e = d["p2"]["effect"]
        base_wk = d["base"].loc[ids].values
        eff = rng.normal(e["log_effect"], e["se"], size=(N_DRAWS, 1))
        per_store = weeks * base_wk * np.expm1(eff)
        baseline_total = float(base_wk.sum() * weeks)
        rob = d["p2"]["robustness"]
        effect_desc = (f"{e['effect_pct']:+.1%} per week (95% CI {e['ci_pct'][0]:+.1%} to "
                       f"{e['ci_pct'][1]:+.1%})")
        method = "Stacked difference-in-differences of Promo2 enrollment (src/sic/causal/promo2_did.py)"
        confidence = "low"
        assumptions += [f"Baseline = average weekly forecast, {d['baseline_weeks']}."]
        if already:
            assumptions.append(f"{len(already)} selected stores are already enrolled and were "
                               "excluded.")
        caveats += [(f"NOT proven: enrolled stores were declining before enrollment (pre-trend "
                    f"p={d['p2']['pretrend_test']['p_value']:.4f}) and the placebo is non-zero; "
                    "part of the estimate may be rebound."),
                    (f"Sensitivity: trend-adjusted {rob['trend_adjusted']['effect_pct']:+.1%}, "
                    f"donut {rob['donut_drop_12w_pre']['effect_pct']:+.1%}."),
                    "Run a randomized rollout before relying on this number."]
        store_ids = ids

    total = per_store.sum(axis=1)
    by_store = pd.DataFrame({"store_id": store_ids, "expected_eur": per_store.mean(axis=0)})
    by_store = by_store.sort_values("expected_eur", key=np.abs, ascending=False)
    res = {
        "scenario": s.scenario, "description": SCENARIOS[s.scenario],
        "selection": selection, "stores": len(store_ids), "weeks": weeks,
        "effect_used": effect_desc,
        "total_eur": _summary(total),
        "per_week_eur": _summary(total / weeks),
        "pct_of_baseline": _summary(total / baseline_total),
        "baseline_eur": baseline_total,
        "baseline_per_week_eur": baseline_total / weeks,  # so the LLM never divides itself
        "baseline_basis": ("non-promo sales of the affected Mon-Fri days only"
                           if s.scenario.endswith("promo_week")
                           else "total forecast sales of the selected stores over the period"),
        "top_stores": by_store.head(10).round(0).to_dict(orient="records"),
        "confidence": confidence, "method": method,
        "assumptions": assumptions, "caveats": caveats,
        "uncertainty": f"90% interval from {N_DRAWS:,} Monte Carlo draws of the causal estimate",
        "inputs": {k: v for k, v in asdict(s).items() if v not in (None, False, [], 7)},
    }
    return res


def run(**kwargs) -> dict:
    return simulate(Scenario(**kwargs))


if __name__ == "__main__":
    import sys

    print(json.dumps(run(**json.loads(sys.argv[1])), indent=1, default=str))
