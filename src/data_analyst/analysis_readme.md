# Stage 1 analysis capabilities

The deterministic analysis layer currently supports:

- Numeric descriptive statistics
- Frequency/value counts
- Grouped numeric summaries using mean, sum, min, max and median
- Grouped row counts
- Dataset profiling
- Generic data-quality reporting
- Read-only SQL against Parquet through DuckDB

These functions are intentionally independent of the NYC Taxi schema so the same engine can later be validated on UDISE+ and World Bank data.
