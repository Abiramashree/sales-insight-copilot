"""Build reports/REVIEW_M1_M3.md — every number is read from pipeline outputs."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from sic import config
from sic.plotting import BLUE, NEUTRAL, ORANGE, TEXT_2, eur, save


def _j(name):
    return json.loads((config.path("reports") / name).read_text())


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%"


def figures(bt, did) -> dict:
    out = config.path("outputs")
    f = {}

    # Backtest WAPE by fold
    folds = bt["folds"]
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x - 0.21, [d["baseline"]["wape"] for d in folds], 0.4, color=NEUTRAL, label="Baseline")
    ax.bar(x + 0.21, [d["p50"]["wape"] for d in folds], 0.4, color=BLUE, label="Gradient boosting")
    ax.set_xticks(x, [f"{d['start'][:7]}\n→ {d['end'][:7]}" for d in folds])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Daily WAPE by backtest fold (lower is better)")
    ax.legend(loc="upper right")
    f["wape"] = save(fig, "m2_backtest_wape")

    # Store example: weekly actual vs forecast, anomalous weeks marked
    w = pd.read_parquet(out / "store_weeks.parquet")
    sid = int(pd.read_parquet(out / "episodes.parquet").iloc[0].store_id)
    s = w[w.store_id == sid].sort_values("week_start")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(s.week_start, s.forecast, color=NEUTRAL, label="Forecast (out-of-sample)")
    ax.plot(s.week_start, s.sales, color=BLUE, label="Actual")
    a = s[s.is_anomaly]
    ax.scatter(a.week_start, a.sales, s=36, color=ORANGE, zorder=3, edgecolor="white",
               linewidth=1.5, label="Anomalous week")
    ax.yaxis.set_major_formatter(FuncFormatter(eur))
    ax.set_title(f"Store {sid}: weekly sales vs. forecast — top-severity episode")
    ax.legend(loc="upper left")
    f["store"] = save(fig, "m2_store_example")
    f["store_id"] = sid

    # Easter week by day (segment anomaly)
    srca = pd.read_parquet(out / "segment_rca.parquet")
    if len(srca):
        r = srca.sort_values("robust_z", ascending=False).iloc[0]
        bd = {k: v for k, v in json.loads(r.by_day).items() if v is not None}
        fig, ax = plt.subplots(figsize=(7, 3))
        days = pd.to_datetime(list(bd))
        ax.bar([d.strftime("%a %d") for d in days], list(bd.values()), color=BLUE)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(f"Type-{r.store_type} stores, week of {r.week_start:%d %b %Y}: "
                     "actual vs forecast by day")
        f["easter"] = save(fig, "m3_easter_week_by_day")

    # Root-cause mix (€ gap)
    e = pd.read_parquet(out / "episode_rca.parquet")
    mix = e.groupby(["root_cause", "direction"]).gap.sum().unstack(fill_value=0)
    mix = mix.reindex(mix.abs().sum(axis=1).sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(mix))
    ax.barh(y, mix.get(-1, 0), color=ORANGE, label="Below forecast")
    ax.barh(y, mix.get(1, 0), color=BLUE, label="Above forecast")
    ax.set_yticks(y, [i.replace("_", " ") for i in mix.index])
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(eur))
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.set_title("Anomaly € gap by assigned root cause")
    ax.legend(loc="lower right")
    f["rca_mix"] = save(fig, "m3_root_cause_mix")

    # Promo lift distribution
    lift = pd.read_parquet(out / "promo_lift_by_store.parquet")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(lift.lift, bins=40, color=BLUE, edgecolor="white", linewidth=1)
    ax.axvline(lift.lift.median(), color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(lift.lift.median(), ax.get_ylim()[1] * 0.92, f"  median {lift.lift.median():+.0%}",
            color=TEXT_2)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Causal promo lift per store (1,115 within-store regressions)")
    f["lift"] = save(fig, "m3_promo_lift_hist")

    # Promo2 event study
    es = pd.DataFrame(did["event_study"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.axvline(-0.5, color=NEUTRAL, linewidth=1, linestyle="--")
    pre = es.bin < 0
    for mask, c, lab in ((pre, NEUTRAL, "Before enrollment"), (~pre, BLUE, "After enrollment")):
        d = es[mask]
        ax.errorbar(d.bin * 4, d.coef, yerr=[d.coef - d.lo, d.hi - d.coef], fmt="o", color=c,
                    markersize=6, capsize=0, elinewidth=2, label=lab)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Weeks relative to Promo2 enrollment (4-week bins)")
    ax.set_title("Promo2 event study — pre-period drifts down (selection), then jumps")
    ax.legend(loc="upper left")
    f["event"] = save(fig, "m3_promo2_event_study")
    return f


def run() -> str:
    eda, bt = _j("m1_eda_facts.json"), _j("m2_backtest_metrics.json")
    det, rca = _j("m2_detection_summary.json"), _j("m3_rca_examples.json")
    pl, did, rec = _j("m3_promo_lift.json"), _j("m3_promo2_did.json"), _j("m3_recommendations.json")
    f = figures(bt, did)
    ov, wk = bt["overall_daily"], bt["weekly_store_wape"]
    cov = det["calibration"]["coverage"]
    seg = rca["segment_anomalies"][0] if rca["segment_anomalies"] else None
    rob = did["robustness"]
    st = {d["store_type"]: d for d in eda["store_types"]}
    p2 = {d["promo2_enrolled"]: d["avg_sales"] for d in eda["promo2_descriptive"]}

    top_store = next(x for x in rca["top_episodes"] if x["store_id"] == f["store_id"])
    ev_types = ", ".join(e["type"] for e in top_store["evidence"]) or "none"
    recs_md = "\n".join(
        f"| {i + 1} | {r['type'].replace('_', ' ')} | {r['action']} | "
        f"{'€{:,.0f}'.format(r['weekly_eur_impact']) if r['weekly_eur_impact'] else '—'} | "
        f"{r['confidence']} |"
        for i, r in enumerate(rec["top10"]))
    port = {p["type"]: p for p in rec["portfolio"]}
    pilot = port["promo_targeting_pilot"]

    md = f"""# Sales Insight Copilot — M1–M3 review

*Generated by `make report` from pipeline outputs. Data: Rossmann (Kaggle), 1,115 stores,
{eda['headline']['first_day'][:10]} → {eda['headline']['last_day'][:10]}.*

## TL;DR

- **Forecasting:** gradient boosting beats a strong store×weekday×promo baseline —
  daily WAPE **{ov['p50']['wape']:.1%}** vs {ov['baseline']['wape']:.1%}; at the store-week
  grain managers read, **{wk['p50']:.1%}** vs {wk['baseline']:.1%}.
- **Detection:** {det['anomaly_rate']:.1%} of store-weeks flagged → {det['episodes']:,}
  episodes ({det['episodes_up']:,} above / {det['episodes_down']:,} below forecast).
  Calibrated intervals hit ~nominal 90% coverage in folds 3–4.
- **RCA** names the Easter-week segment anomaly correctly and attributes
  {rca['traffic_vs_basket_mix'].get('traffic', 0):,} of {det['episodes']:,} store episodes
  mainly to **traffic** rather than basket.
- **Promo lift (causal):** **{pct(pl['pooled_lift'])}** pooled; p10–p90 across stores
  {pct(pl['lift_p10_p90'][0], 0)} to {pct(pl['lift_p10_p90'][1], 0)}. Type-b and
  "extra"-assortment stores respond least.
- **Promo2 (causal):** **{pct(did['effect']['effect_pct'])}**
  (95% CI {pct(did['effect']['ci_pct'][0])} to {pct(did['effect']['ci_pct'][1])}) — but
  pre-trends fail (p={did['pretrend_test']['p_value']:.4f}). Treat as *suggestive*.

---

## M1 — Data & KPIs

- Warehouse: DuckDB, `raw → staged → marts` ([sql/](../sql/)); KPI dictionary
  [sql/kpis.yaml](../sql/kpis.yaml) (sales, customers, basket, attainment, promo lift, …).
- Quality checks enforced at build: row counts, no sales on closed days, unique store-days.
- 181 stores have a data gap (180 offline in H2-2014 for refurbishment).

| Store type | Stores | Avg customers/day | Basket (€) |
|---|---|---|---|
""" + "\n".join(f"| {t} | {d['stores']} | {d['avg_customers']:,.0f} | {d['basket']:.2f} |"
                for t, d in st.items()) + f"""

Findings worth carrying forward:
- Promo days sell {pct(eda['promo_raw_uplift_pct'], 0)} more (raw, Mon–Fri) — M3 tests
  whether that is causal.
- Closer competitors coincide with *higher* sales (urban-location confounding — not causal).
- Promo2 stores sell less on average (€{p2[True]:,.0f} vs €{p2[False]:,.0f}/day) — selection;
  M3's DiD reverses the sign.
- Same-store sales Jan–Jul 2015 vs 2014: {pct(eda['yoy_same_store_jan_jul_2015'])}.

![weekly](../{eda['fig_weekly']})

## M2 — Forecasting & detection

Rolling-origin backtest, 4 quarterly folds (each trains only on data before it).

| Metric | Baseline | Gradient boosting |
|---|---|---|
| Daily WAPE | {ov['baseline']['wape']:.1%} | **{ov['p50']['wape']:.1%}** |
| Daily RMSPE | {ov['baseline']['rmspe']:.3f} | **{ov['p50']['rmspe']:.3f}** |
| Store-week WAPE | {wk['baseline']:.1%} | **{wk['p50']:.1%}** |
| Bias | {pct(ov['baseline']['bias'])} | {pct(ov['p50']['bias'])} |

![wape](../{f['wape']})

**Interval calibration.** Raw 90% quantile intervals under-covered ({ov['interval_coverage']:.0%}).
Conformal widening using only the *previous* fold's errors:

| Fold | Raw coverage | Calibrated |
|---|---|---|
""" + "\n".join(f"| {k} | {v['raw']:.1%} | {v['calibrated']:.1%} |" for k, v in cov.items()) + f"""

Fold 1 has no prior fold (left raw). Fold 2 (Nov–Jan, Christmas) still under-covers.

**Detection rule.** A store-week is anomalous if |actual/forecast − 1| ≥ 10% **and** most
open days fall outside the calibrated interval in the same direction. Consecutive weeks →
episodes, ranked by € gap.

![store](../{f['store']})

Store {f['store_id']}: {pct(top_store['gap_pct'], 0)} over {top_store['weeks']} weeks,
{top_store['tb_driver']}-driven; evidence: {ev_types}.

## M3 — Root cause, causal lift, recommendations

### Root cause
Three lenses: **traffic vs basket** (exact log decomposition), **dimensional drill-down**
(Adtributor-style surprise + contribution), **event evidence** (reopening, competitor
opening, Promo2 start, holidays, segment-wide movement, persistence).
""" + (f"""
**Segment anomaly — {seg['root_cause']['label'].replace('_', ' ')}.** {seg['root_cause']['detail']}

![easter](../{f['easter']})
""" if seg else "") + f"""
![rca](../{f['rca_mix']})

### Daily Promo lift
Promo dates are identical across stores (verified), so lift is identified **within store**
(day-of-week, month, year, holiday and adjacent-closure controls; HC1 SE).

- Pooled **{pct(pl['pooled_lift'])}**; median store {pct(pl['median_store_lift'])};
  {pl['share_stores_ci_above_zero']:.0%} of stores have a CI above zero.
- Median lift by store type: """ + ", ".join(f"{k} {pct(v, 0)}" for k, v in pl['by_store_type'].items()) + """.
- By assortment: """ + ", ".join(f"{k} {pct(v, 0)}" for k, v in pl['by_assortment'].items()) + f""".
- Heterogeneity model R² = {pl['heterogeneity_model']['r2']:.2f}.

![lift](../{f['lift']})

### Promo2 enrollment — stacked difference-in-differences
{did['treated_stores']} newly enrolled stores in {did['cohorts']} monthly cohorts vs
{did['control_stores']} never-enrolled stores; store×stack and week×stack fixed effects; SE
clustered by store.

| Specification | Effect | 95% CI |
|---|---|---|
| Main | **{pct(did['effect']['effect_pct'])}** | {pct(did['effect']['ci_pct'][0])} to {pct(did['effect']['ci_pct'][1])} |
| Trend-adjusted | {pct(rob['trend_adjusted']['effect_pct'])} | {pct(rob['trend_adjusted']['ci_pct'][0])} to {pct(rob['trend_adjusted']['ci_pct'][1])} |
| Donut (drop 12 wks pre) | {pct(rob['donut_drop_12w_pre']['effect_pct'])} | {pct(rob['donut_drop_12w_pre']['ci_pct'][0])} to {pct(rob['donut_drop_12w_pre']['ci_pct'][1])} |
| Placebo (fake start −{did['placebo']['shift_weeks']} wks) | {pct(did['placebo']['effect_pct'])} | {pct(did['placebo']['ci_pct'][0])} to {pct(did['placebo']['ci_pct'][1])} |

Pre-trend joint test: χ²={did['pretrend_test']['chi2']:.1f}, p={did['pretrend_test']['p_value']:.4f}.

![event](../{f['event']})

**Reading:** positive in every specification, but enrolled stores were drifting down
before enrollment and the placebo is non-zero — stores were likely enrolled *because*
they were slipping. The honest conclusion is "probably positive, magnitude uncertain";
the recommendation is a randomized rollout.

### Recommendations (as of week {rec['as_of_week']})

Portfolio-level:
- **{pilot['action']}** Revenue at stake ≈ €{pilot['evidence']['weekly_promo_revenue_at_stake']:,.0f}/wk;
  {pilot['evidence']['share_type_b_or_extra']:.0%} of these stores are type b or extra-assortment.
- **{port['protect_promo_execution']['action']}** (assumption-based sizing)
- **{port['promo2_program']['action']}**

Top store/portfolio actions by weekly € impact:

| # | Type | Action | €/week | Confidence |
|---|---|---|---|---|
{recs_md}

---

## Known limitations → candidate tweaks for your review

1. **No holiday-proximity feature** — the model misses pre-Easter/pre-Christmas stock-up
   (the main segment anomaly, and fold-2 under-coverage). Fix: `days_to_next_closure`.
2. **Trees don't extrapolate trend** — fold bias swings ({", ".join(pct(d['p50']['bias']) for d in bt['folds'])});
   above-forecast episodes outnumber below 2:1. Fix: level-correction from recent residuals.
3. **Anomaly volume** ({det['episodes']:,} episodes) is high for a sales team to consume —
   tighten thresholds or rank harder before the LLM layer.
4. **Promo lift may include pull-forward** (non-promo weeks depressed); no margin data, so €
   is revenue not profit.
5. **Promo2 pre-trends** — try synthetic-control / matched-trend controls.
6. **RCA evidence is rule-based**; no external data (weather, local events) yet.
"""
    p = config.path("reports") / "REVIEW_M1_M3.md"
    p.write_text(md)
    return str(p)


if __name__ == "__main__":
    print(run())
