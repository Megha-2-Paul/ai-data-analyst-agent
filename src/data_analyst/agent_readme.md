# Stage 1.4 — Agent Orchestration

Stage 1.4 adds the first agent layer on top of the Stage 1 analytical engine.

The important architectural boundary is:

```text
User question
    ↓
Planner
    ↓
Validated QueryPlan
    ↓
Agent executor
    ↓
Analytical engine
    ↓
Structured result + execution trace
```

The agent does **not** perform numerical calculations itself. It orchestrates the
existing analytical functions and returns the plan and execution steps alongside
the result.

This stage intentionally keeps the planner dependency injectable. A future LLM
adapter can replace the natural-language planning implementation while retaining
the same `QueryPlan` contract and executor. That prevents the model from
directly executing arbitrary Python or SQL.

Example:

```python
from data_analyst.agent import AnalystAgent

response = AnalystAgent().ask(
    "What is the average fare_amount by payment_type?",
    dataframe,
)

print(response.to_dict())
```

This provides the foundation for a later LLM-backed analyst without coupling
the core analytical engine to a model provider or API key.
