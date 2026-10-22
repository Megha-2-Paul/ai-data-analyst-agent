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
from .answer import AnalystAnswer, generate_answer
from .planner import QueryPlan, plan_query
from .profiling import profile_dataset
from .quality import quality_report
from .reasoning import AnalysisPlan, execute_analysis_plan, plan_analysis
from .serialization import json_safe
from .ml import detect_anomalies


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
    answer: AnalystAnswer | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "question": self.question,
            "plan": self.plan.to_dict(),
            "result": self.result,
            "execution_steps": self.execution_steps,
        }
        if self.analysis_plan is not None:
            payload["analysis_plan"] = self.analysis_plan.to_dict()
        if self.answer is not None:
            payload["answer"] = self.answer.to_dict()
        return json_safe(payload)


def _execute_plan(df: pl.DataFrame, plan: QueryPlan) -> Any:
    if plan.filter_column:
        if plan.filter_column not in df.columns:
            raise AgentExecutionError(f"Filter column not found: {plan.filter_column!r}")
        if plan.filter_value is None:
            raise AgentExecutionError("Filter column is set but filter value is missing.")
        value = plan.filter_value
        matches = df.get_column(plan.filter_column).cast(pl.String).str.to_lowercase() == str(value).lower()
        df = df.filter(matches)

    if plan.operation == "anomaly_detection":
        try:
            return detect_anomalies(df, plan.columns or None)
        except RuntimeError as exc:
            raise AgentExecutionError(str(exc)) from exc
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
        if plan.group_by and not plan.metric:
            if plan.aggregation == "count":
                result = {
                    "analysis": "grouped_count",
                    "parameters": {
                        "group_by": plan.group_by,
                        "metric": "count",
                        "aggregation": "count",
                    },
                    "result": df.group_by(plan.group_by, maintain_order=True).len(name="count").to_dicts(),
                    "metadata": {"row_count": df.height},
                }
                if plan.sort_direction:
                    result["parameters"]["sort_direction"] = plan.sort_direction
                    result["result"].sort(
                        key=lambda row: row["count"],
                        reverse=plan.sort_direction == "desc",
                    )
                if plan.limit:
                    result["parameters"]["limit"] = plan.limit
                if plan.percentage_of_total:
                    denominator = sum(float(row["count"]) for row in result["result"])
                    for row in result["result"]:
                        row["percentage_of_total"] = (
                            float(row["count"]) / denominator * 100.0 if denominator else None
                        )
                    result["parameters"]["percentage_of_total"] = True
                    result["metadata"]["percentage_basis"] = "count"
                    result["metadata"]["percentage_denominator"] = denominator
                if plan.limit:
                    result["result"] = result["result"][: plan.limit]
                return result
            raise AgentExecutionError("Aggregate plan is missing metric.")
        result = aggregate(
            df,
            group_by=plan.group_by or None,
            metric=plan.metric,
            agg=plan.aggregation or "mean",
        )
        if plan.sort_direction:
            result["parameters"]["sort_direction"] = plan.sort_direction
        if plan.limit:
            result["parameters"]["limit"] = plan.limit
        if plan.sort_direction and isinstance(result.get("result"), list):
            rows = result["result"]
            # Sort only non-null values in the requested direction, then append
            # null aggregates. Using reverse=True on a tuple key would also
            # reverse the null placement and could rank missing values first.
            non_null_rows = [row for row in rows if row.get(plan.metric) is not None]
            null_rows = [row for row in rows if row.get(plan.metric) is None]
            non_null_rows.sort(
                key=lambda row: row[plan.metric],
                reverse=plan.sort_direction == "desc",
            )
            rows = non_null_rows + null_rows
            result["result"] = rows
        if plan.percentage_of_total and plan.group_by and isinstance(result.get("result"), list):
            value_key = plan.metric or "row_count"
            rows = result["result"]
            values = [row.get(value_key) for row in rows]
            valid_values = [float(value) for value in values if value is not None]
            denominator = sum(valid_values)
            for row in rows:
                value = row.get(value_key)
                row["percentage_of_total"] = (
                    float(value) / denominator * 100.0
                    if value is not None and denominator != 0 else None
                )
            result["parameters"]["percentage_of_total"] = True
            result["metadata"]["percentage_basis"] = value_key
            result["metadata"]["percentage_denominator"] = denominator
        if plan.limit and isinstance(result.get("result"), list):
            result["result"] = result["result"][: plan.limit]
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
        # Supply a bounded vocabulary of observed non-numeric values so the
        # deterministic planner can resolve natural phrasing such as "for the
        # West region" without hardcoding any domain-specific schema.
        categorical_values: dict[str, list[Any]] = {}
        for name, dtype in df.schema.items():
            if dtype.is_numeric() or dtype in (pl.Date, pl.Datetime):
                continue
            values = (
                df.get_column(name)
                .drop_nulls()
                .unique(maintain_order=True)
                .head(250)
                .to_list()
            )
            categorical_values[name] = [
                value for value in values
                if len(str(value)) <= 120
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
                answer=generate_answer(result),
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

        planner_kwargs = {
            "columns": df.columns,
            "numeric_columns": numeric_columns,
            "datetime_columns": datetime_columns,
        }
        # Preserve compatibility with injected planners (including LLM planners)
        # whose call signatures predate the optional observed-value vocabulary.
        if self._planner is plan_query:
            planner_kwargs["categorical_values"] = categorical_values

        plan = self._planner(question, **planner_kwargs)
        result = _execute_plan(df, plan)

        return AgentResponse(
            question=question,
            plan=plan,
            result=result,
            answer=generate_answer(result),
            execution_steps=[
                "inspect_dataset_schema",
                f"plan:{plan.operation}",
                "validate_plan",
                "execute_analysis",
                "return_structured_result",
            ],
        )
