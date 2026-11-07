# App spec (agreed 2026-11-06)

Two pages. Numbers always come from deterministic code; the LLM only writes and converses.

## 1. KPIs
- Scope: whole chain, or one store (searchable). No store-type level.
- Period: week, month or quarter; pick the period.
- Numbers: sales, vs forecast, vs last year (day-matched). Nothing else.
- Trend chart for the scope, with the selected period highlighted.
- Anomalies: top 10 by € gap in the period (chain: store-weeks; store: its weeks).
  Clicking one shows:
  - Why: likely cause, customers vs basket, evidence.
  - Context: the store's weeks before/after, and how look-alike stores did that week.
  - Next step: the pipeline's recommended action, plus a button that opens the
    Assistant prefilled with "Investigate store X, week of Y".
  - No daily table.

## 2. Assistant (junior BA)
- One chat for ad-hoc questions and initiative planning. Goals in plain words; it asks
  clarifying questions and refines the plan in conversation.
- A plan (LLM-written around deterministic numbers) always has:
  plausibility (Plausible / Stretch / Unlikely with numbers), stores and the % each must
  grow, if/then/because hypotheses each tested on history, an A/B design (stores per
  group, weeks, detectable lift, reproducible test/control list), and alternatives sized
  in € with a range.
- A/B readout: dry run on history (no fake effect) or uploaded CSV of real results.
- Sources collapsed under every answer; unverified numbers flagged.
- Manual panel beside the chat (no LLM): test a hypothesis, design an A/B, analyze results.

## Not in the UI
Live feed controls (code stays: `make feed-start`), the old Pipeline/Actions/Test/Ask pages.
