"""Agent orchestration for the analytical engine.

Stage 1.4 keeps language interpretation and numerical computation separate.
A planner creates a validated QueryPlan; this module executes that plan and
returns a traceable AgentResponse. An LLM can be plugged in at the planning
boundary later without changing the analytical execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from .analysis import aggregate, correlation, describe, time_series
from .planner import QueryPlan, plan_query
from .profiling import profile_dataset
from .quality import quality_report
from .reasoning import AnalysisPlan, execute_analysis_plan, plan_analysis


class AgentExecutionError(RuntimeError):
    """Raised when a valid plan cannot be executed."""


@dataclass
class AgentResponse:
    """Traceable response produced by the agent orchestration layer."""

    question: str
    plan: QueryPlan
    result: Any
    execution_steps: list[str] = field(default_factory=list)
    analysis_plan: AnalysisPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "question": self.question,
            "plan": self.plan.to_dict(),
            "result": self.result,
            "execution_steps": self.execution_steps,
        }
        if self.analysis_plan is not None:
            payload["analysis_plan"] = self.analysis_plan.to_dict()
        return payload


def _execute_plan(df: pl.DataFrame, plan: QueryPlan) -> Any:
    if plan.operation == "profile":
        return profile_dataset(df)
    if plan.operation == "quality":
        return quality_report(df)
    if plan.operation == "describe":
        return describe(df, plan.columns)
    if plan.operation == "correlation":
        return correlation(df, plan.columns)
    if plan.operation == "time_series":
        if not plan.datetime_column:
            raise AgentExecutionError("Time-series plan is missing datetime_column.")
        return time_series(
            df,
            datetime_column=plan.datetime_column,
            frequency=plan.frequency or "1d",
            metric=plan.metric,
            agg=plan.aggregation or "count",
        )
    if plan.operation == "aggregate":
        if not plan.group_by:
            raise AgentExecutionError("Aggregate plan is missing group_by.")
        if not plan.metric:
            if plan.aggregation == "count":
                return {
                    "analysis": "grouped_count",
                    "parameters": {
                        "group_by": plan.group_by,
                        "aggregation": "count",
                    },
                    "result": df.group_by(plan.group_by, maintain_order=True).len(name="count").to_dicts(),
                    "metadata": {"row_count": df.height},
                }
            raise AgentExecutionError("Aggregate plan is missing metric.")
        result = aggregate(
            df,
            group_by=plan.group_by,
            metric=plan.metric,
            agg=plan.aggregation or "mean",
        )
        if plan.sort_direction and isinstance(result.get("result"), list):
            rows = result["result"]
            rows.sort(
                key=lambda row: (row.get(plan.metric) is None, row.get(plan.metric)),
                reverse=plan.sort_direction == "desc",
            )
            if plan.limit:
                result["result"] = rows[: plan.limit]
        return result
    raise AgentExecutionError(f"Unsupported operation: {plan.operation}")


class AnalystAgent:
    """Small deterministic agent core built around the validated planner."""

    def __init__(self, planner: Callable[..., QueryPlan] = plan_query) -> None:
        self._planner = planner

    def ask(
        self,
        question: str,
        df: pl.DataFrame,
    ) -> AgentResponse:
        numeric_columns = [
            name for name, dtype in df.schema.items() if dtype.is_numeric()
        ]
        datetime_columns = [
            name for name, dtype in df.schema.items()
            if dtype in (pl.Date, pl.Datetime)
        ]

        try:
            analysis_plan = plan_analysis(
                question,
                columns=df.columns,
                numeric_columns=numeric_columns,
                datetime_columns=datetime_columns,
            )
        except Exception as exc:
            # Stage 1 questions continue through the existing single-step planner.
            from .planner import PlanningError
            if not isinstance(exc, PlanningError):
                raise
        else:
            result = execute_analysis_plan(df, analysis_plan, _execute_plan)
            return AgentResponse(
                question=question,
                plan=analysis_plan.steps[0].plan,
                result=result,
                execution_steps=[
                    "inspect_dataset_schema",
                    "plan:multi_step_reasoning",
                    *[f"validate:{step.step_id}" for step in analysis_plan.steps],
                    *[f"execute:{step.step_id}" for step in analysis_plan.steps],
                    "combine_evidence",
                    "return_structured_result",
                ],
                analysis_plan=analysis_plan,
            )

        plan = self._planner(
            question,
            columns=df.columns,
            numeric_columns=numeric_columns,
            datetime_columns=datetime_columns,
        )
        result = _execute_plan(df, plan)

        return AgentResponse(
            question=question,
            plan=plan,
            result=result,
            execution_steps=[
                "inspect_dataset_schema",
                f"plan:{plan.operation}",
                "validate_plan",
                "execute_analysis",
                "return_structured_result",
            ],
        )
