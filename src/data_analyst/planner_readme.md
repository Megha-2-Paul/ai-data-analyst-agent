# Stage 1.3 — Query Planning

Stage 1.3 introduces a conservative query-planning layer between a user's
analytical question and the Stage 1 analytical engine.

The planner converts supported natural-language requests into a validated
`QueryPlan`. It does not execute analysis and it never invents dataset
columns.

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

Ambiguous requests raise `PlanningError` rather than silently guessing.
