# AI Data Analyst Agent

An extensible AI-powered data analysis platform designed to analyze real-world structured datasets using a reliable analytical engine, with an agent layer added incrementally.

## Project vision

The goal is to build an analyst that can work across different real-world datasets without hard-coded, dataset-specific analysis logic.

Initial real-world validation sources:

- NYC TLC Yellow Taxi Trip Records — primary Stage 1 dataset
- UDISE+ education data — later validation dataset
- World Bank World Development Indicators — later API integration

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
- Optional LLM-backed query planning
- Agent orchestration and execution traces
- Multi-step analytical reasoning with dependency-aware execution
- Automated tests
- Command-line interface

### Stage 1.5 — Optional LLM Planner

The deterministic planner remains the default. Stage 1.5 adds an optional OpenAI planner at the planning boundary:

~~~text
User question
    ↓
Planner (deterministic OR optional OpenAI)
    ↓
Validated QueryPlan
    ↓
Agent executor
    ↓
Analytical engine
    ↓
Structured result + execution trace
~~~

The LLM only proposes a structured QueryPlan. It does not execute Python, SQL, or arbitrary tools. Every referenced column, operation, aggregation, datetime field, frequency, sort direction, and limit is validated before the analytical engine runs.

### Zero-cost-by-default

The base installation does not require the OpenAI SDK or an API key. CI tests the LLM boundary with a fake client, so tests make no network requests and do not consume API credits.

Install the optional dependency only when live LLM planning is wanted:

~~~bash
pip install -e ".[llm]"
~~~

Then configure credentials in the environment:

~~~bash
export OPENAI_API_KEY="..."
~~~

Optionally choose a model:

~~~bash
export OPENAI_MODEL="gpt-5.6-luna"
~~~

Live API usage is separate from the repository's free CI/test path.

CLI:

~~~bash
# Default: deterministic planner, no API call
data-analyst ask data/yellow_tripdata_2025-01.parquet "What is the average fare_amount by payment_type?"

# Optional: use the OpenAI planner
data-analyst ask data/yellow_tripdata_2025-01.parquet "What is the average fare_amount by payment_type?" --planner openai
~~~

### Stage 2 — Multi-step Analytical Reasoning

Complex analytical questions can now be decomposed into a validated dependency graph. A later step may consume a value selected by an earlier step without allowing the model to execute arbitrary code.

Example:

~~~text
Which payment type has the highest average fare,
and did that payment type's average fare increase over the year?

Step 1: average fare by payment type → select highest
Step 2: filter to selected payment type → analyze fare over time
Step 3: compare first vs last value → summarize direction
~~~

Each step records its plan, dependencies, applied filter binding, and result. A safety cap limits a plan to eight steps, dependency cycles are rejected, and execution remains inside the existing analytical engine.

The Stage 2 `ask` flow automatically detects the supported multi-step pattern; ordinary Stage 1 questions continue through the existing single-step planner.

### Stage 3 — Evidence & Answer Generation

The agent now turns structured analytical results into deterministic, evidence-backed analyst answers.

Each generated finding carries one or more evidence IDs. Evidence is extracted only from values already returned by the analytical engine, so Stage 3 does not introduce a second LLM generation call or invent statistics. Unsupported result types are reported explicitly rather than guessed.

The answer model contains:

- concise summary
- findings with evidence references
- structured evidence items with source step IDs and values
- explicit limitations when results are insufficient or unsupported

For multi-step analysis, Stage 3 links the selected group from the first step to the first-to-last trend from the final step.

The `ask` CLI now prints the analyst-style answer first, followed by the complete structured response.

### Stage 1.4 — Agent Orchestration

The agent connects the question → planner → analytical engine pipeline. The agent does not perform numerical calculations itself; it orchestrates existing analytical tools.

The project deliberately does not commit external datasets to the repository. See docs/data_sources.md for source and download guidance.

## Planned architecture

~~~text
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
Evidence extraction + validation
    ↓
Analyst answer + evidence trace
    ↓
[Next] Visualizations
~~~

## Technology

- Python
- Polars
- DuckDB
- NumPy
- SciPy
- Pytest
- Optional OpenAI SDK for live LLM planning

## Development policy

This repository is being developed incrementally. Each stage is validated before the next layer is introduced. External data remains attributed to its original publisher.

## Status

**Stage 3 — Evidence & Answer Generation: implemented**
