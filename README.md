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
- Automated tests
- Command-line interface

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
Analytical tools
    ├── statistics
    ├── grouped analysis
    └── DuckDB queries
    ↓
Structured analysis results
    ↓
[Stage 2+] AI Analyst Agent
    ↓
[Stage 3+] Visualizations + evidence-backed insights
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

**Stage 1 — Foundation: in development**
