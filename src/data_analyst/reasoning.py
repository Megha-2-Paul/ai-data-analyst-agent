"""Multi-step analytical planning and execution for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import polars as pl

from .planner import PlanningError, QueryPlan


class ReasoningError(PlanningError):
    """Raised when a multi-step analytical plan is unsafe or invalid."""


MAX_STEPS = 8


@dataclass(frozen=True)
class AnalysisStep:
    """One validated analytical step and its optional dependency binding."""

    step_id: str
    plan: QueryPlan
    depends_on: list[str] = field(default_factory=list)
    filter_from_step: str | None = None
    filter_source_column: str | None = None
    filter_target_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "plan": self.plan.to_dict(),
            "depends_on": list(self.depends_on),
            "filter_from_step": self.filter_from_step,
            "filter_source_column": self.filter_source_column,
            "filter_target_column": self.filter_target_column,
        }


@dataclass(frozen=True)
class AnalysisPlan:
    """Validated DAG of analytical steps."""

    question: str
    steps: list[AnalysisStep]
    final_step_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "steps": [step.to_dict() for step in self.steps],
            "final_step_id": self.final_step_id,
        }


def _validate_steps(steps: Sequence[AnalysisStep], final_step_id: str) -> None:
    if not steps:
        raise ReasoningError("Analysis plan must contain at least one step.")
    if len(steps) > MAX_STEPS:
        raise ReasoningError(f"Analysis plan cannot contain more than {MAX_STEPS} steps.")

    ids = [step.step_id for step in steps]
    if len(ids) != len(set(ids)):
        raise ReasoningError("Analysis step IDs must be unique.")
    known = set(ids)

    if final_step_id not in known:
        raise ReasoningError(f"Final step does not exist: {final_step_id!r}")

    graph = {step.step_id: set(step.depends_on) for step in steps}
    for step in steps:
        if step.step_id in step.depends_on:
            raise ReasoningError(f"Step cannot depend on itself: {step.step_id!r}")
        unknown = set(step.depends_on) - known
        if unknown:
            raise ReasoningError(
                f"Step {step.step_id!r} references unknown dependencies: {sorted(unknown)}"
            )
        if step.filter_from_step:
            if step.filter_from_step not in step.depends_on:
                raise ReasoningError(
                    f"Filter source {step.filter_from_step!r} must be a dependency of {step.step_id!r}."
                )
            if not step.filter_source_column or not step.filter_target_column:
                raise ReasoningError(
                    f"Step {step.step_id!r} requires both filter source and target columns."
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ReasoningError("Analysis plan contains a circular dependency.")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def validate_analysis_plan(plan: AnalysisPlan) -> AnalysisPlan:
    _validate_steps(plan.steps, plan.final_step_id)
    return plan


def _schema_columns(df: pl.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
    datetime = [name for name, dtype in df.schema.items() if dtype in (pl.Date, pl.Datetime)]
    return df.columns, numeric, datetime


def _find_column(question: str, candidates: Sequence[str]) -> str | None:
    normalized = question.lower().replace("_", " ")
    matches = [c for c in candidates if c.lower().replace("_", " ") in normalized]
    return sorted(matches, key=len, reverse=True)[0] if matches else None


def plan_analysis(
    question: str,
    *,
    columns: Sequence[str],
    numeric_columns: Sequence[str] | None = None,
    datetime_columns: Sequence[str] | None = None,
) -> AnalysisPlan:
    """Recognize a conservative two-step 'identify then analyze over time' request.

    Other questions intentionally raise PlanningError so Stage 1 behavior remains
    unchanged and the normal QueryPlan planner handles them.
    """
    if not question or not question.strip():
        raise ReasoningError("Question cannot be empty.")

    text = question.lower()
    if not any(marker in text for marker in (" and did ", " and has ", " and was ")):
        raise ReasoningError("Question is not a recognized multi-step analytical request.")
    if not any(marker in text for marker in ("over the year", "over time", "over the month", "over the period")):
        raise ReasoningError("Multi-step request must contain an explicit time comparison.")

    group_column = _find_column(question, columns)
    metric = _find_column(question, numeric_columns or [])
    datetime_column = _find_column(question, datetime_columns or [])
    if not group_column or not metric or not datetime_column:
        raise ReasoningError(
            "Multi-step request requires an explicitly named grouping, numeric metric, and datetime column."
        )

    aggregation = "mean" if any(word in text for word in ("average", "avg", "mean")) else None
    if aggregation is None:
        raise ReasoningError("The current multi-step planner requires an average/mean metric.")

    if not any(word in text for word in ("highest", "largest", "maximum", "top")):
        raise ReasoningError("The current multi-step planner requires a highest/top first-step comparison.")

    if any(word in text for word in ("monthly", "month")):
        frequency = "1mo"
    elif any(word in text for word in ("weekly", "week")):
        frequency = "1w"
    else:
        frequency = "1d"

    step1 = AnalysisStep(
        step_id="step_1",
        plan=QueryPlan(
            operation="aggregate",
            group_by=[group_column],
            metric=metric,
            aggregation=aggregation,
            sort_direction="desc",
            limit=1,
            question=question,
            assumptions=["Select the group with the highest average metric."],
        ),
    )
    step2 = AnalysisStep(
        step_id="step_2",
        plan=QueryPlan(
            operation="time_series",
            datetime_column=datetime_column,
            metric=metric,
            aggregation=aggregation,
            frequency=frequency,
            question=question,
            assumptions=["Analyze the metric over time for the group identified in step_1."],
        ),
        depends_on=["step_1"],
        filter_from_step="step_1",
        filter_source_column=group_column,
        filter_target_column=group_column,
    )

    plan = AnalysisPlan(question=question, steps=[step1, step2], final_step_id="step_2")
    return validate_analysis_plan(plan)


def _extract_filter_value(source_result: Any, source_column: str) -> Any:
    rows = source_result.get("result") if isinstance(source_result, dict) else source_result
    if not isinstance(rows, list) or not rows:
        raise ReasoningError("Dependency result does not contain a non-empty row list.")
    row = rows[0]
    if not isinstance(row, dict) or source_column not in row:
        raise ReasoningError(
            f"Dependency result does not contain required column {source_column!r}."
        )
    value = row[source_column]
    if value is None:
        raise ReasoningError(f"Dependency result selected a null value for {source_column!r}.")
    return value


def execute_analysis_plan(
    df: pl.DataFrame,
    plan: AnalysisPlan,
    execute_plan: Callable[[pl.DataFrame, QueryPlan], Any],
) -> dict[str, Any]:
    """Execute a validated plan in dependency order and retain evidence for every step."""
    validate_analysis_plan(plan)

    steps_by_id = {step.step_id: step for step in plan.steps}
    results: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    completed: set[str] = set()

    while len(completed) < len(plan.steps):
        progressed = False
        for step in plan.steps:
            if step.step_id in completed:
                continue
            if not set(step.depends_on).issubset(completed):
                continue

            working_df = df
            filter_value = None
            if step.filter_from_step:
                source = steps_by_id[step.filter_from_step]
                filter_value = _extract_filter_value(
                    results[source.step_id],
                    step.filter_source_column or "",
                )
                target = step.filter_target_column or ""
                if target not in working_df.columns:
                    raise ReasoningError(f"Filter target column not found: {target!r}")
                working_df = working_df.filter(pl.col(target) == filter_value)

            result = execute_plan(working_df, step.plan)
            results[step.step_id] = result
            traces.append({
                "step_id": step.step_id,
                "depends_on": list(step.depends_on),
                "filter": (
                    {
                        "from_step": step.filter_from_step,
                        "source_column": step.filter_source_column,
                        "target_column": step.filter_target_column,
                        "value": filter_value,
                    }
                    if step.filter_from_step
                    else None
                ),
                "plan": step.plan.to_dict(),
                "result": result,
            })
            completed.add(step.step_id)
            progressed = True

        if not progressed:
            raise ReasoningError("Unable to resolve analysis step dependencies.")

    return {
        "analysis": "multi_step_reasoning",
        "question": plan.question,
        "steps": traces,
        "final_step_id": plan.final_step_id,
        "final_result": results[plan.final_step_id],
    }
