# AI Data Analyst Agent

An extensible AI-powered data analysis platform designed to analyze **real-world structured datasets** using a reliable analytical engine and an agent layer.

## Project vision

The goal is to build an analyst that can work across different real-world **structured/tabular datasets** without hard-coded, dataset-specific analysis logic.

Initial real-world validation sources:

- **NYC TLC Yellow Taxi Trip Records** — primary validation dataset
- **UDISE+ education data** from India's Open Government Data platform — later validation dataset
- **World Bank World Development Indicators** — later validation dataset/API integration

## Analytical pipeline

```text
Dataset
  ↓
Ingestion
  ↓
Profiling + Quality Assessment
  ↓
Safe Data Preparation
  ↓
Validated Analytical Plan
  ↓
Analytical Engine
  ↓
Evidence
  ↓
Answer
```

## Stage 4 — Data Preparation

Stage 4 adds a dataset-agnostic preparation layer that runs **before analysis when explicitly requested**.

### Safe automatic transformations

- Normalize column names to stable snake_case names
- Trim leading/trailing whitespace from string values
- Convert blank string values to null
- Parse obvious datetime columns when at least 95% of their non-null values parse successfully
- Remove completely empty columns

### Recommendations rather than silent mutation

The preparation layer detects issues that require statistical or domain judgment but does **not** silently change the data:

- Missing-value imputation
- Row deletion because of missing values
- Outlier removal
- Exact duplicate removal
- Domain-specific validity corrections

Every preparation run produces:

- Original profile
- Original quality report
- Cleaning plan
- Applied transformation trace
- Recommendations
- Prepared profile
- Prepared quality report

The original dataframe is never mutated.

### CLI

Prepare a dataset and inspect the complete preparation report:

```powershell
data-analyst prepare data/raw/yellow_tripdata_2025-01.parquet
```

Apply safe preparation before answering a question:

```powershell
data-analyst ask data/raw/yellow_tripdata_2025-01.parquet "What is the average fare_amount by payment_type?" --prepare
```

Use `--json` with `ask` for the complete structured response.

## Earlier stages

### Stage 1 — Foundation

- CSV and Parquet ingestion
- Dataset profiling
- Data-quality checks
- Descriptive statistics
- Grouped/time-based analysis
- DuckDB analytical queries
- Structured, traceable analysis results
- Automated tests
- CLI

### Stage 1.3 — Query Planning

Natural-language questions are converted into validated `QueryPlan` objects without inventing dataset columns.

### Stage 1.4 — Agent Orchestration

The agent executes validated plans through the analytical engine and records an execution trace.

### Stage 1.5 — Optional LLM Planner

An OpenAI planner can be enabled explicitly. The deterministic planner remains the default, so the project is zero-cost by default.

### Stage 2 — Multi-step Analytical Reasoning

The agent can execute supported dependency-aware multi-step questions and preserve evidence from each step.

### Stage 3 — Evidence & Answer Generation

Answers are generated deterministically from analytical results. Findings reference structured evidence IDs, and unsupported result types produce explicit limitations.

## Technology

- Python
- Polars
- DuckDB
- NumPy
- SciPy
- Pytest

## Development policy

This repository is developed incrementally. Each stage is validated before the next layer is introduced. External datasets remain attributed to their original publishers, and raw datasets are never overwritten by preparation.

## Status

**Stage 4 — Data Preparation: implementation in progress**
