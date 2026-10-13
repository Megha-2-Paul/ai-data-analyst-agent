# Stage 1.3 — Query Planning

Stage 1.3 introduces a conservative query-planning layer between a user's
analytical question and the Stage 1 analytical engine.

The planner converts supported natural-language requests into a validated
`QueryPlan`. Neither `plan_query` nor the `plan` CLI command executes analysis,
and the planner never invents dataset columns.

Supported intents:

- dataset profiling
- data-quality checks
- descriptive statistics
- grouped aggregations
- Pearson correlation
- time-series analysis

This plan schema is intentionally suitable as a future LLM tool-call contract.
The LLM will eventually select or populate the same operations, while the
analytical engine remains responsible for computation.

Example:

```python
from data_analyst.planner import plan_query

plan = plan_query(
    "What is the average fare_amount by payment_type?",
    columns=["payment_type", "fare_amount"],
    numeric_columns=["fare_amount"],
)
print(plan.to_dict())
```

Ambiguous requests raise `PlanningError` rather than silently guessing. This
includes multiple candidate metrics or datetime columns and conflicting
aggregation or time-frequency terms. Hour/hourly, day/daily, week/weekly,
month/monthly, and year/yearly/annual/annually map to the corresponding Polars
intervals. Explicit quarterly requests raise `PlanningError`; they are not
silently changed to daily. Frequency words inside a matched schema column name
are not treated as frequency instructions. When no time frequency or
aggregation is specified, the plan records the daily (`1d`) and count defaults
in `assumptions`.

An explicit supported frequency is sufficient to express time-series intent;
the question does not also need to contain the word `trend`.

## Analysis API relationship

`agg`, `group_by`, `metric`, `datetime_column`, and `frequency` use the
corresponding analysis API parameter names. Serialized plans keep `aggregation`
as a compatibility alias for `agg`. A grouped count with no metric means row
count; `aggregate` supports that case with `metric=None` and `agg="count"`.
The `sort_direction` and `limit` fields preserve requested result shaping, but
the current analysis functions do not accept those parameters. A future
execution layer must apply them separately; plans are not direct function-call
arguments and this stage does not provide that execution layer.
