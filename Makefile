RUN = uv run python -m

.PHONY: setup data eda forecast select-features detect rca causal recommend opportunity kpis report all test lint

setup:      ## install deps into .venv
	uv sync

data:       ## M1: build DuckDB warehouse (raw -> staged -> marts)
	$(RUN) sic.data.build

eda:        ## M1: exploratory facts + figures
	$(RUN) sic.eda.explore

forecast:   ## M2: rolling-origin backtest (~7 min)
	$(RUN) sic.forecast.backtest

select-features: ## M2: feature selection on pre-backtest windows (~30 min) -> reports/m2_feature_selection.json
	$(RUN) sic.forecast.select

detect:     ## M2: interval calibration + anomaly episodes
	$(RUN) sic.detect.anomalies

rca:        ## M3: root-cause analysis
	$(RUN) sic.rca.engine

causal:     ## M3: promo lift + Promo2 DiD + competitor-opening impact
	$(RUN) sic.causal.promo_lift
	$(RUN) sic.causal.promo2_did
	$(RUN) sic.causal.competitor_impact

recommend:  ## M3: ranked recommendations
	$(RUN) sic.causal.recommend

opportunity: ## store opportunity scores -> data/outputs/opportunity_scores.parquet
	$(RUN) sic.opportunity.score

kpis:       ## weekly KPI scorecard -> reports/weekly_kpis.md
	$(RUN) sic.kpi.weekly

report:     ## build reports/REVIEW_M1_M3.md
	$(RUN) sic.report.build

all: data eda forecast detect rca causal recommend opportunity kpis report

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

# ---------- M4: LLM layer (needs the provider key, GROQ_API_KEY) ----------
.PHONY: llm-check cards ask chat

llm-check:  ## verify provider key / model id with one tiny call
	$(RUN) sic.llm.client

cards:      ## M4: weekly insight cards -> reports/insight_cards.md
	$(RUN) sic.insights.cards

ask:        ## M4: one question, e.g. make ask Q="Why is store 921 down?"
	$(RUN) sic.agent.chat "$(Q)"

chat:       ## M4: interactive agent
	$(RUN) sic.agent.chat

llm-probe:  ## compare reasoning_effort settings: latency, tokens, truncation
	$(RUN) sic.llm.client --probe

demo:       ## live cards if the LLM is healthy, else the saved snapshot
	$(RUN) sic.insights.demo

demo-cached: ## show saved demo cards (no network)
	$(RUN) sic.insights.demo --cached

app:        ## Streamlit UI -> http://localhost:8501
	uv run streamlit run app/streamlit_app.py

# ---------- M5: evals & monitoring ----------
.PHONY: eval eval-live eval-report

eval:       ## offline evals: retrieval, card QA gate, A/B design, drift, observability
	$(RUN) sic.evals.retrieval_eval > /dev/null
	$(RUN) sic.evals.playbook_eval > /dev/null
	$(RUN) sic.evals.card_qa_eval > /dev/null
	$(RUN) sic.evals.ab_design > /dev/null
	$(RUN) sic.evals.drift > /dev/null
	$(RUN) sic.evals.observability > /dev/null
	$(RUN) sic.evals.report

eval-live:  ## live-LLM evals: agent golden set + LLM critic on injected defects
	$(RUN) sic.evals.agent_eval
	$(RUN) sic.evals.card_qa_eval --llm
	$(RUN) sic.evals.report

eval-report:
	$(RUN) sic.evals.report
