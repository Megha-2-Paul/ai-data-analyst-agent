# AI Data Analyst Agent

An extensible AI-powered data analysis platform designed to analyze **real-world structured datasets** using a reliable analytical engine, with an agent layer added in later stages.

## Project vision

The goal is to build an analyst that can work across different real-world datasets without hard-coded, dataset-specific analysis logic.

Initial real-world validation sources:

- **NYC TLC Yellow Taxi Trip Records** — primary Stage 1 dataset
- **UDISE+ education data** from India's Open Government Data platform — later validation dataset
- **World Bank World Development Indicators** — later API integration

## Stage 1 — Foundation

Stage 1 focuses on the trustworthy analytical core rather than the LLM agent itself:

- CSV and Parquet ingestion
- Dataset profiling
- Data-quality checks
- Descriptive statistics
- Grouped/time-based analysis
- DuckDB analytical queries
- Structured, traceable analysis results
- Natural-language query planning
- Automated tests
- Command-line interface

### Stage 1.3 — Query Planning (Complete)

Stage 1.3 is complete and merged into GitHub `main` by PR #10. The deterministic planner converts supported natural-language requests into validated `QueryPlan` objects using only columns supplied by the dataset schema. It rejects ambiguous or unsupported requests, recognizes supported time frequencies, and records defaults when a frequency is omitted.

The planner plans; it does not execute analysis. The `plan` CLI command returns a JSON plan. The analytical engine remains responsible for validation and computation.

The Stage 1.3 fix commit is `3a704c4a4f77b645eaaea92e7d458668f67783a6`; the GitHub merge commit is `0e322ace92e28e6dc945a5c5081a2bf8a27602ac`.

### Stage 1.4 — Optional Groq Integration (Planned)

Stage 1.4 is planned, not implemented. Groq may be added as an optional aid for interpreting questions, proposing query plans, and formulating explanations grounded in actual analytical results. The existing Python engine must remain authoritative for schema validation, execution, and numerical results. AI must not invent columns or data, and deterministic operation without an AI service must remain available.

API keys will be kept outside source control. The integration will need explicit handling for unavailable services, malformed responses, rate limits, and other API errors. Stage 1.4 requires its own approved plan, implementation, and test cycle; this repository currently has no Groq integration or LLM agent.

The project deliberately does **not** commit external datasets to the repository. See [`docs/data_sources.md`](docs/data_sources.md) for source and download guidance.

## Architecture and roadmap

```text
Real-world dataset
    ↓
Data ingestion
    ↓
Dataset profiling
    ↓
Data-quality engine
    ↓
Deterministic query planning (implemented)
    ↓
Deterministic analytical tools
    ├── statistics
    ├── grouped analysis
    ├── time-series analysis
    └── DuckDB queries
    ↓
Structured analysis results
    ↓
[Stage 1.4, planned] Optional Groq-assisted interpretation and explanations
    ↓
[Later, planned] Visualizations and evidence-backed insights
```

Only the ingestion, profiling, quality, analysis, result, CLI, and deterministic planning layers are implemented in the current Stage 1 foundation. Groq, an AI agent, visualization, and evidence-grounded explanation features are future work.

## Technology

- Python
- Polars
- DuckDB
- NumPy
- SciPy
- Pytest

## Development policy

This repository is being developed incrementally. Each stage is validated before the next layer is introduced. External data remains attributed to its original publisher. See [Development Workflow](docs/WORKFLOW.md) for the collaboration, approval, Git safety, and validation process.

## Status

**Stage 1.3 — Query Planning: complete and merged into `main`**

**Stage 1.4 — Optional Groq Integration: planned, not implemented**
