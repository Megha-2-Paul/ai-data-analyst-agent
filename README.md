# AI Data Analyst Agent

An extensible AI-powered data analysis platform designed to analyze **real-world structured datasets** using a reliable analytical engine, with an agent layer added incrementally.

## Project vision

The goal is to build an analyst that can work across different real-world datasets without hard-coded, dataset-specific analysis logic.

Initial real-world validation sources:

- **NYC TLC Yellow Taxi Trip Records** — primary Stage 1 dataset
- **UDISE+ education data** from India's Open Government Data platform — later validation dataset
- **World Bank World Development Indicators** — later API integration

## Stage 1 — Foundation

Stage 1 builds a trustworthy analytical core and the first agent orchestration layer:

- CSV and Parquet ingestion
- Dataset profiling
- Data-quality checks
- Descriptive statistics
- Grouped/time-based analysis
- DuckDB analytical queries
- Structured, traceable analysis results
- Natural-language query planning
- Agent orchestration and execution traces
- Automated tests
- Command-line interface

### Stage 1.3 — Query Planning

The query-planning layer converts supported natural-language analytical requests into validated `QueryPlan` objects. It only references columns supplied by the dataset schema and raises an explicit planning error when a request is ambiguous or unsupported.

### Stage 1.4 — Agent Orchestration

The agent layer now connects the question → planner → analytical engine pipeline.

```text
User question
    ↓
Query planner
    ↓
Validated QueryPlan
    ↓
Agent executor
    ↓
Analytical engine
    ↓
Structured result + execution trace
```

The agent does not perform numerical calculations itself. It orchestrates the existing analytical tools. The planner is dependency-injected so a future LLM can replace the deterministic planner without changing the execution layer.

CLI example:

```bash
data-analyst ask data/yellow_tripdata_2025-01.parquet "What is the average fare_amount by payment_type?"
```

The project deliberately does **not** commit external datasets to the repository. See [`docs/data_sources.md`](docs/data_sources.md) for source and download guidance.

## Planned architecture

```text
Real dataset
    ↓
Data ingestion
    ↓
Dataset profiling
    ↓
Data-quality engine
    ↓
Query planning
    ↓
Agent orchestration
    ↓
Analytical tools
    ├── statistics
    ├── grouped analysis
    ├── time-series analysis
    └── DuckDB queries
    ↓
Structured analysis results
    ↓
[Next] LLM-backed planning
    ↓
[Later] Visualizations + evidence-backed insights
```

## Technology

- Python
- Polars
- DuckDB
- NumPy
- SciPy
- Pytest

## Development policy

This repository is being developed incrementally. Each stage is validated before the next layer is introduced. External data remains attributed to its original publisher.

## Status

**Stage 1.4 — Agent Orchestration: in development**
