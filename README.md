# Sales Insight Copilot

An automated sales insight pipeline: it **detects** what changed, **diagnoses** why, and
**recommends** what to do, with every number traceable to the data. An LLM insight layer
(weekly cards and a tool-using chat agent) comes next and will narrate these outputs without
computing numbers itself.

Built on the public Rossmann Store Sales dataset (1,115 stores, daily, 2013–2015).

```
CSV ─▶ DuckDB warehouse ─▶ forecasts ─▶ anomalies ─▶ root cause ─▶ causal lift ─▶ recommendations
       raw/staged/marts    (backtested)  (calibrated)  (3 lenses)   (FE + DiD)     (ranked, € sized)
       + KPI dictionary                                                              │
                                                              M4: LLM cards + agent ◀┘
```

## Status

| Milestone | What | Status |
|---|---|---|
| M1 | Warehouse, KPI dictionary, EDA | ✅ |
| M2 | Forecasting backtest, interval calibration, anomaly detection | ✅ |
| M3 | Root-cause engine, promo lift, Promo2 DiD, recommendations | ✅ |
| M4 | LLM insight cards + tool-using chat agent (Groq · gpt-oss-120b), KPI RAG | ✅ built, awaiting live run |
| M5 | Evals: retrieval, card-QA defect injection, agent golden set, A/B power, drift, LLM observability | ✅ |
| M6 | Streamlit app ✅ · FastAPI, Docker, CI | in progress |

**Results and limitations:** see [reports/REVIEW_M1_M3.md](reports/REVIEW_M1_M3.md).
Weekly scorecard: [reports/weekly_kpis.md](reports/weekly_kpis.md).
Evals & monitoring: [reports/M5_EVALS.md](reports/M5_EVALS.md) (`make eval`, `make eval-live`).

### Forecast model v2 (holiday proximity, promo sequence, recent store level)

| Metric (rolling-origin backtest) | Baseline | v1 | **v2** |
|---|---|---|---|
| Daily WAPE | 12.6% | 9.7% | **8.7%** |
| Store-week WAPE | 9.5% | 7.1% | **6.0%** |
| Christmas fold WAPE | 16.3% | 12.5% | **10.7%** |

v2 removed the pre-Easter false anomaly and cut below-forecast episodes from 750 to 525.
Known trade-off: a slight summer-2015 under-forecast (fold 4 bias −2.9%).

## Run it

```bash
# 1. Download train.csv + store.csv from kaggle.com/c/rossmann-store-sales
#    and set paths.raw_source in config.yaml
make setup
make all       # ~8 min, mostly the forecast backtest
make test
```

### LLM layer (M4)

Works with any OpenAI-compatible provider. The default is [Groq](https://console.groq.com)
running `openai/gpt-oss-120b` with low reasoning effort. Provider, model and prices are set in
`config.yaml → llm`.

```bash
export GROQ_API_KEY=gsk_...      # or put it in .env (gitignored)
make llm-check                   # one tiny call: confirms key + model id
make cards                       # weekly insight cards -> reports/insight_cards.md
make ask Q="Why is store 921 below forecast?"
make chat                        # interactive
make demo                        # live cards, or saved snapshot if the LLM is down/slow
make demo-cached                 # saved snapshot only (no network)
make app                         # Streamlit UI at http://localhost:8501
```

**Demo safety net.** A card run where at least 80% of cards pass QA is saved to
`demo/insight_cards_snapshot.json`. A weaker run can never overwrite it. When the live
LLM fails, is rate-limited, or goes over its time budget, the CLI and the app show a
short notice and then display the saved cards.

How the LLM layer stays honest:
- **Fact packs.** The LLM only receives numbered, pre-formatted facts taken from the pipeline,
  and it must cite them (`[f3]`).
- **Numeric fact-check.** Every number in a card or answer must match a source value. Direction
  errors (e.g. "+23%" for a −23% fact) count as failures.
- **Critic and revision.** An LLM reviewer flags unsupported claims and overstated certainty.
  A failing card gets one rewrite, and if it still fails it's held back, not published.
- **SQL sandbox.** The agent queries in-memory DuckDB with the warehouse attached read-only,
  external file access turned off, and the configuration locked.
- **Logged calls.** Every call goes to `data/outputs/llm_calls.jsonl` (latency, tokens, cost),
  and agent runs go to `agent_traces.jsonl`. These are the inputs for M5 evals.

### Competitive-response playbooks (governed RAG)

The copilot only suggests actions that come from **approved store playbooks**
([docs/playbooks/playbooks.yaml](docs/playbooks/playbooks.yaml)). They were written for this
project and are not real Rossmann policy.

- **Governance:** only `approved`, in-date, latest-version playbooks can be retrieved. Drafts,
  superseded versions and expired guidance are blocked, and an eval checks for leaks.
- **Measured impact:** a stacked difference-in-differences study of 120 competitor openings
  (`make causal`) estimates **−15.4%** weekly sales when a competitor opens within 1 km and −5.4%
  beyond that. Pre-trends are flat and the placebo is zero. Playbooks quote these numbers
  straight from the analysis.
- **Where playbooks are used:** the root-cause engine picks a playbook by matching its evidence
  (for example, store 770 had a competitor open 100 m away, so it gets `PB-COMP-NEAR v2`).
  Insight-card actions must be one of that playbook's plays and cite it, which code enforces.
  The agent has a `get_playbook` tool, and the store explorer shows the matching playbook.

### What-if simulator (causal inference + AI)

Ask *"What if a competitor opens 300 m from store 262?"* or *"What if we add a promo week in
the 20 lowest-response stores?"*. The work is split between the LLM and the models:

- **The LLM translates.** The copilot maps the question onto a supported scenario and store
  selection by calling the `simulate_scenario` tool, then explains the result.
- **Causal estimates compute.** [src/sic/whatif/engine.py](src/sic/whatif/engine.py) applies
  effects measured earlier in the pipeline to each store's baseline: per-store promo lift,
  competitor-opening impact (within or beyond 1 km), and Promo2 (marked low confidence). It
  returns the expected € impact with a **90% range from Monte Carlo draws**, plus the method,
  assumptions and caveats.
- **Scenarios the engine doesn't support** are refused, and the copilot lists the ones it can
  run.
- **A form in the app** (Ask the Copilot → What-if simulator) runs the same engine with **zero
  LLM tokens**.

**Show your work.** Every chat answer has an expandable panel listing each tool call with its
SQL or parameters, the data returned (as tables, simulation charts or playbook citations), and
which numbers in the answer were verified against tool outputs.

## Layout

| Path | Purpose |
|---|---|
| `sql/` | Warehouse SQL (`01_raw` → `02_staged` → `03_marts`) and `kpis.yaml` |
| `src/sic/data` | Warehouse build + data-quality checks |
| `src/sic/eda` | Exploratory facts and figures |
| `src/sic/forecast` | Features (leakage-safe target encodings), rolling-origin backtest |
| `src/sic/detect` | Conformal interval calibration, store-week anomalies, episodes, segment anomalies |
| `src/sic/rca` | Traffic-vs-basket decomposition, Adtributor-style drill-down, event evidence |
| `src/sic/causal` | FE-OLS helper, within-store promo lift, stacked DiD for Promo2, recommendations |
| `src/sic/report` | Builds the review report from outputs |
| `src/sic/llm` | OpenAI-compatible client (Groq) with call logging; `FakeLLM` for tests |
| `src/sic/insights` | Fact packs, numeric fact-checker, insight cards (write → critique → revise) |
| `src/sic/agent` | Chat agent loop, sandboxed tools, TF-IDF retrieval over KPI + method docs |
| `docs/knowledge/` | Methodology and promo-program docs the agent retrieves from |
| `src/sic/kpi` | Weekly KPI scorecard (day-matched WoW/YoY, same-store) |
| `app/` | Streamlit UI: KPIs, insight cards, store explorer, chat, model |
| `demo/` | Saved insight-card snapshot used when the live LLM is unavailable |
| `docs/playbooks/` | Governed store-response playbooks (status, version, review date, triggers) |
| `src/sic/playbooks` | Playbook retrieval: governance filter, evidence matching, semantic search |
| `src/sic/whatif` | What-if scenario engine: causal effects x baselines, Monte Carlo ranges |
| `config.yaml` | Paths, backtest folds, detection thresholds, causal windows |

## Method notes

- **Forecasting:** a global `HistGradientBoostingRegressor` on log sales, with quantile models
  for the intervals. It's compared against a store × weekday × promo baseline over 4 quarterly
  rolling-origin folds. Store target encodings are computed only from each fold's training window.
- **Calibration:** conformalized quantile regression. Each fold's interval is widened by the
  previous fold's 90th-percentile conformity score, so calibration only ever uses past data.
- **Promo lift:** promo dates are the same for every store, so lift is estimated within each store
  over time, with day-of-week, month, year, holiday and adjacent-closure controls.
- **Promo2:** stores enroll in Promo2 at different times, so the analysis is a stacked
  difference-in-differences (cohort stacks, never-enrolled controls, store×stack and week×stack
  fixed effects, SE clustered by store). It includes an event study, a placebo test and
  checks against the pre-enrollment dip.
