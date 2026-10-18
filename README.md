# AI Data Analyst Agent

An extensible data analysis platform with a deterministic analytical engine, optional Groq/Gemini/OpenAI planning, evidence-backed answers, safe data preparation, and an authenticated FastAPI backend.

## Project vision

Build an analyst that can work across real-world structured datasets without hard-coded, dataset-specific analysis logic.

Initial validation sources:

- **NYC TLC Yellow Taxi Trip Records** — primary Stage 1 dataset
- **UDISE+ education data** — later validation dataset
- **World Bank World Development Indicators** — later validation dataset and API integration

External datasets are not committed to this repository. See [data source guidance](docs/data_sources.md).

## Implemented capabilities

- CSV and Parquet ingestion and schema scanning
- Dataset profiling and data-quality checks
- Descriptive statistics, grouped aggregation, correlation, and time-series analysis
- Deterministic natural-language query planning and validation
- Optional OpenAI-backed planning at the planner boundary
- Multi-step analytical reasoning with dependency-aware execution
- Structured execution traces and evidence-backed answers
- Optional safe data preparation with a transformation trace
- Command-line interface and automated tests
- FastAPI endpoints for authentication, dataset upload, analysis, reports, and usage
- PostgreSQL persistence with Alembic migrations, workspace membership and resource isolation
- Optional Groq and Gemini planner adapters, plus opt-in live provider smoke tests

The Python analytical engine remains authoritative for schema validation, calculations, and numerical results. AI-assisted planning proposes a structured plan; it does not execute arbitrary Python or SQL.

## Stage 1.3 — Deterministic Query Planning

The deterministic planner converts supported natural-language requests into validated `QueryPlan` objects using the supplied dataset schema. It handles supported aggregation, grouping, ranking, filters, and time frequencies, and rejects ambiguous or unsupported requests. When a supported default is used, it should be explicit in the plan.

The planner plans; it does not execute analysis. The `data-analyst plan` command returns a JSON plan, while the analytical engine validates and computes results.

## Stage 1.5 — Optional LLM Planner

The deterministic planner remains the default. An optional OpenAI planner can propose a structured `QueryPlan`; the application validates the plan before execution.

Install the optional dependency only when live LLM planning is wanted:

~~~bash
pip install -e ".[llm]"
~~~

Set credentials through the environment; never commit API keys:

~~~bash
export OPENAI_API_KEY="..."
# Optional: set OPENAI_MODEL to a model available to your account.
~~~

Live API usage may incur provider charges. CI tests the planner boundary without requiring live API calls.

Example commands:

~~~bash
# Deterministic planning/analysis
data-analyst ask data/raw/yellow_tripdata_2025-01.parquet "What is the average fare by payment type?"

# Optional OpenAI-backed planning
data-analyst ask data/raw/yellow_tripdata_2025-01.parquet "What is the average fare by payment type?" --planner openai
~~~

## Stage 2 — Multi-step Analytical Reasoning

Supported complex questions can be decomposed into a validated dependency graph. Later steps can consume selected values from earlier results without allowing the model to execute arbitrary code. Each step records its plan, dependencies, filter bindings, and result. The executor applies a step limit and rejects dependency cycles.

## Stage 3 — Evidence & Answer Generation

The agent turns structured analytical results into deterministic, evidence-backed answers. Findings reference evidence IDs drawn from results produced by the analytical engine; unsupported result types are identified rather than guessed.

The structured answer includes a concise summary, findings with evidence references, evidence items with source step IDs and values, and limitations when results are insufficient or unsupported. Use `--json` with `ask` to inspect the complete response.

## Stage 4 — Data Preparation

Stage 4 adds an optional dataset-agnostic preparation layer that runs before analysis when explicitly requested.

Safe automatic transformations include:

- Normalize column names to stable snake_case names
- Trim leading and trailing whitespace from strings
- Convert blank strings to null
- Parse obvious datetime columns when at least 95% of non-null values parse successfully
- Drop completely empty columns

Statistical or domain-dependent issues are reported as recommendations rather than silently changed. Examples include missing-value imputation, row deletion, outlier removal, exact duplicate removal, and domain-specific validity corrections.

A preparation run retains the original profile and quality report, validated cleaning plan, applied transformation trace, recommendations, prepared profile, and prepared quality report. The source dataframe is not mutated or overwritten.

~~~powershell
# Inspect preparation recommendations and applied safe transformations
data-analyst prepare data/raw/yellow_tripdata_2025-01.parquet

# Prepare the data before answering a question
data-analyst ask data/raw/yellow_tripdata_2025-01.parquet "What is the average fare by payment type?" --prepare

# Inspect the complete structured response
data-analyst ask data/raw/yellow_tripdata_2025-01.parquet "What is the average fare by payment type?" --prepare --json
~~~

## Planned work

Broader real-world validation, additional visualizations, asynchronous workers, and a frontend remain future work. Groq and Gemini planning are optional; credentials remain outside source control, provider failures are controlled, and deterministic validation/calculation remain authoritative.

## Architecture

~~~text
Real-world dataset
    ↓
Ingestion + schema scan
    ↓
Profiling + data-quality checks
    ↓
Optional safe data preparation
    ↓
Deterministic planner OR optional LLM planner
    ↓
Validated QueryPlan
    ↓
Single-step or multi-step agent execution
    ↓
Deterministic analytical engine
    ↓
Structured results + execution trace
    ↓
Evidence extraction + validation
    ↓
Analyst answer + evidence trace
~~~

## Technology

- Python
- Polars
- DuckDB
- NumPy
- SciPy
- Pytest
- Optional OpenAI, Groq and Gemini SDKs for AI-assisted planning

## Development workflow

This repository is developed incrementally. Review the [Development Workflow](docs/WORKFLOW.md) before making changes; it describes the plan-first approval process, Git safety, and validation expectations. Each stage should be reviewed and tested before the next stage is introduced.

## Status

- **Stage 1.3 — Deterministic query planning:** implemented
- **Stage 1.5 — Optional OpenAI planner:** implemented
- **Stage 2 — Multi-step analytical reasoning:** implemented
- **Stage 3 — Evidence-backed answers:** implemented
- **Stage 4 — Data preparation:** implemented on `stage-4-data-preparation`; not merged into `main`, under final validation and review
- **Groq and Gemini providers:** optional adapters with offline tests and opt-in live smoke tests
- **Backend API:** implemented on `stage-4-data-preparation`; requires database migration and environment configuration before use


## Backend API

The authenticated FastAPI backend exposes dataset upload, profile/quality/preparation, analysis, report download, and usage endpoints. See [Backend API documentation](docs/API.md) for setup and route details.

## Groq and Gemini

The deterministic planner is the default. Optional Groq and Gemini planners can be selected with `--planner groq` or `--planner gemini`; configure API keys only through environment variables. See [provider and live-test instructions](docs/PROVIDERS.md). Standard CI uses fake responses and does not consume API quota; live smoke tests run only when provider keys are configured.
