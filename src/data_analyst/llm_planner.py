"""Optional LLM-backed query planning.

The LLM is isolated behind the same QueryPlan contract used by the deterministic
planner. The feature is opt-in: importing this module does not require the
OpenAI package or an API key, and CI can test it with a fake client.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from .planner import PlanningError, QueryPlan


class LLMPlanningError(PlanningError):
    """Raised when an LLM response cannot be converted into a safe plan."""


_QUERY_PLAN_FIELDS = {
    "operation", "columns", "group_by", "metric", "aggregation",
    "datetime_column", "frequency", "sort_direction", "limit", "assumptions",
    "filter_column", "filter_value",
}
_ALLOWED_OPERATIONS = {"profile", "quality", "describe", "aggregate", "correlation", "time_series"}
_ALLOWED_AGGREGATIONS = {"count", "sum", "mean", "median", "min", "max", "std"}
_ALLOWED_FREQUENCIES = {"1h", "1d", "1w", "1mo", "1y"}
_ALLOWED_SORTS = {"asc", "desc"}
_MAX_LIMIT = 1000


def _schema_context(columns, numeric_columns, datetime_columns) -> str:
    return json.dumps({
        "columns": list(columns),
        "numeric_columns": list(numeric_columns or []),
        "datetime_columns": list(datetime_columns or []),
    }, ensure_ascii=False)


def _validate_plan_payload(payload, *, question, columns, numeric_columns, datetime_columns) -> QueryPlan:
    if not isinstance(payload, dict):
        raise LLMPlanningError("LLM response must be a JSON object.")

    unknown = set(payload) - _QUERY_PLAN_FIELDS
    if unknown:
        raise LLMPlanningError(f"LLM response contains unsupported fields: {sorted(unknown)}")

    operation = payload.get("operation")
    if operation not in _ALLOWED_OPERATIONS:
        raise LLMPlanningError(f"Unsupported operation from LLM: {operation!r}")

    available = set(columns)
    numeric = set(numeric_columns or [])
    datetimes = set(datetime_columns or [])

    selected_columns = payload.get("columns") or []
    group_by = payload.get("group_by") or []
    if not isinstance(selected_columns, list) or not isinstance(group_by, list):
        raise LLMPlanningError("columns and group_by must be arrays.")

    for field_name, values in (("columns", selected_columns), ("group_by", group_by)):
        if not all(isinstance(name, str) and name for name in values):
            raise LLMPlanningError(f"{field_name} must contain only non-empty strings.")
        if len(values) != len(set(values)):
            raise LLMPlanningError(f"{field_name} must not contain duplicate columns.")
        for name in values:
            if name not in available:
                raise LLMPlanningError(f"LLM referenced unknown column: {name!r}")

    metric = payload.get("metric")
    if metric is not None:
        if metric not in available:
            raise LLMPlanningError(f"LLM referenced unknown metric: {metric!r}")
        if metric not in numeric:
            raise LLMPlanningError(f"LLM metric must be numeric: {metric!r}")

    filter_column = payload.get("filter_column")
    filter_value = payload.get("filter_value")
    if filter_column is not None:
        if filter_column not in available:
            raise LLMPlanningError(f"LLM referenced unknown filter column: {filter_column!r}")
        if not isinstance(filter_column, str) or not filter_column:
            raise LLMPlanningError("filter_column must be a non-empty string.")
        if filter_value is None or not isinstance(filter_value, (str, int, float, bool)):
            raise LLMPlanningError("filter_value must be a simple scalar when filter_column is provided.")

    datetime_column = payload.get("datetime_column")
    if datetime_column is not None:
        if datetime_column not in available:
            raise LLMPlanningError(f"LLM referenced unknown datetime column: {datetime_column!r}")
        if datetime_column not in datetimes and not (
            datetime_column in numeric and datetime_column == "year"
        ):
            raise LLMPlanningError(
                f"LLM datetime_column must be a datetime column or integer year column: {datetime_column!r}"
            )

    aggregation = payload.get("aggregation")
    if aggregation is not None and aggregation not in _ALLOWED_AGGREGATIONS:
        raise LLMPlanningError(f"Unsupported aggregation: {aggregation!r}")

    frequency = payload.get("frequency")
    if frequency is not None and frequency not in _ALLOWED_FREQUENCIES:
        raise LLMPlanningError(f"Unsupported frequency: {frequency!r}")

    sort_direction = payload.get("sort_direction")
    if sort_direction is not None and sort_direction not in _ALLOWED_SORTS:
        raise LLMPlanningError(f"Unsupported sort direction: {sort_direction!r}")

    limit = payload.get("limit")
    if limit is not None and (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= _MAX_LIMIT
    ):
        raise LLMPlanningError(f"limit must be an integer between 1 and {_MAX_LIMIT}.")

    assumptions = payload.get("assumptions") or []
    if not isinstance(assumptions, list) or not all(isinstance(item, str) and item.strip() for item in assumptions):
        raise LLMPlanningError("assumptions must be an array of non-empty strings.")

    if operation == "correlation":
        if len(selected_columns) < 2:
            raise LLMPlanningError("Correlation requires at least two columns.")
        if any(name not in numeric for name in selected_columns):
            raise LLMPlanningError("Correlation columns must all be numeric.")
    elif operation == "describe":
        if not selected_columns:
            raise LLMPlanningError("Describe requires at least one column.")
        if any(name not in numeric for name in selected_columns):
            raise LLMPlanningError("Describe columns must all be numeric.")
    elif operation == "aggregate":
        if not group_by:
            raise LLMPlanningError("Aggregate requires at least one group_by column.")
        if aggregation is None:
            raise LLMPlanningError("Aggregate requires an aggregation.")
        if aggregation != "count" and not metric:
            raise LLMPlanningError("Aggregate requires a metric unless aggregation is count.")
    elif operation == "time_series":
        if not datetime_column:
            raise LLMPlanningError("Time-series analysis requires datetime_column.")
        if aggregation is None:
            raise LLMPlanningError("Time-series analysis requires an aggregation.")
        if aggregation != "count" and not metric:
            raise LLMPlanningError("Time-series analysis requires a metric unless aggregation is count.")
        if frequency is None:
            raise LLMPlanningError("Time-series analysis requires a frequency.")

    return QueryPlan(
        operation=operation,
        columns=selected_columns,
        group_by=group_by,
        metric=metric,
        aggregation=aggregation,
        datetime_column=datetime_column,
        frequency=frequency,
        sort_direction=sort_direction,
        limit=limit,
        question=question,
        assumptions=assumptions,
        filter_column=filter_column,
        filter_value=filter_value,
    )


class OpenAIPlanner:
    """Optional OpenAI-backed planner; never called unless explicitly used."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMPlanningError("OPENAI_API_KEY is not configured.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMPlanningError(
                "OpenAI planner requires the optional 'llm' dependency. "
                "Install with: pip install -e '.[llm]'"
            ) from exc
        self._client = OpenAI()
        return self._client

    def __call__(
        self,
        question: str,
        *,
        columns: Sequence[str],
        numeric_columns: Sequence[str] | None = None,
        datetime_columns: Sequence[str] | None = None,
    ) -> QueryPlan:
        client = self._get_client()
        system = (
            "You are a conservative data-analysis query planner. "
            "Return ONLY a JSON object matching the requested fields. "
            "Never invent column names. Choose only supported operations."
        )
        user = (
            f"User question: {question}\n"
            f"Dataset schema: {_schema_context(columns, numeric_columns, datetime_columns)}\n"
            "Allowed operations: profile, quality, describe, aggregate, correlation, time_series.\n"
            "Allowed aggregations: count, sum, mean, median, min, max, std.\n"
            "Allowed frequencies: 1h, 1d, 1w, 1mo.\n"
            "Return JSON fields: operation, columns, group_by, metric, aggregation, "
            "datetime_column, frequency, sort_direction, limit, assumptions, filter_column, filter_value."
        )
        try:
            response = client.responses.create(
                model=self._model,
                instructions=system,
                input=user,
            )
            payload = json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise LLMPlanningError("LLM returned invalid JSON.") from exc
        except Exception as exc:
            raise LLMPlanningError(f"LLM planning request failed: {exc}") from exc

        return _validate_plan_payload(
            payload,
            question=question,
            columns=columns,
            numeric_columns=numeric_columns,
            datetime_columns=datetime_columns,
        )
