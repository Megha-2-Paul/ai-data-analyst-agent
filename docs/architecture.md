# Stage 1 architecture

```text
Real dataset
    |
    v
Data ingestion (CSV / Parquet)
    |
    v
Dataset profiling
    |
    v
Data-quality checks
    |
    v
Analytical tools
  |-- descriptive statistics
  |-- value counts
  |-- grouped summaries
  `-- read-only DuckDB SQL
    |
    v
Structured AnalysisResult
    |
    v
Stage 2+: AI Analyst Agent
```

## Design principles

1. **Dataset agnostic:** analysis functions operate on schemas supplied at runtime rather than knowing the NYC Taxi schema.
2. **Real data first:** external data comes from original publishers and is never replaced with invented examples for the portfolio demonstration.
3. **Safe analytical execution:** Stage 1 DuckDB queries are restricted to read-only SELECT/WITH statements.
4. **Traceability:** `AnalysisResult` records the operation, inputs, filters, calculation, result, warnings and metadata.
5. **Scalability:** Parquet can be scanned lazily with Polars and queried directly with DuckDB instead of requiring every source to be eagerly loaded into memory.
6. **Incremental delivery:** the LLM/agent layer is intentionally separated from the deterministic analytical core.
