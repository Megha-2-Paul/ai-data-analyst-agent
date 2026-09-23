"""Natural-language query planning for the Stage 1 analytical engine.

The planner is deliberately conservative: it converts common analytical requests
into a structured plan, but it does not execute analysis or invent columns.
The future LLM agent can use the same plan schema as a tool-call contract.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

Operation = Literal[
    "profile",
    "quality",
    "describe",
    "aggregate",
    "correlation",
    "time_series",
]


class PlanningError(ValueError):
    """Raised when a question cannot be mapped safely to an analysis plan."""


@dataclass(frozen=True)
class QueryPlan:
    """Validated, execution-ready description of one analytical request."""

    operation: Operation
    columns: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    metric: str | None = None
    aggregation: str | None = None
    datetime_column: str | None = None
    frequency: str | None = None
    sort_direction: str | None = None
    limit: int | None = None
    question: str = ""
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_AGGREGATIONS = {
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "sum": "sum",
    "total": "sum",
    "median": "median",
    "minimum": "min",
    "min": "min",
    "maximum": "max",
    "max": "max",
    "count": "count",
    "number": "count",
    "standard deviation": "std",
    "std": "std",
}

def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _find_columns(question: str, columns: Sequence[str]) -> list[str]:
    normalized_question = f" {_normalize(question)} "
    matches = []
    for column in columns:
        candidate = _normalize(column)
        if candidate and f" {candidate} " in normalized_question:
            matches.append(column)
    # Preserve the order in which columns are named in the question.
    # This keeps structured plans deterministic and avoids changing the user's
    # requested correlation column order.
    positions = {
        column: normalized_question.find(f" {_normalize(column)} ")
        for column in matches
    }
    # Prefer the more specific column when names overlap, such as
    # gdp and gdp_growth. Both occur at the same position in
    # "average GDP growth", but gdp_growth is the intended metric.
    return sorted(
        matches,
        key=lambda c: (positions[c], -len(_normalize(c))),
    )


def _numeric_columns(question: str, columns: Sequence[str], numeric_columns: Sequence[str] | None) -> list[str]:
    candidates = list(columns if numeric_columns is None else numeric_columns)
    return [c for c in _find_columns(question, candidates)]


def _datetime_columns(question: str, columns: Sequence[str], datetime_columns: Sequence[str] | None) -> list[str]:
    candidates = list(columns if datetime_columns is None else datetime_columns)
    return [c for c in _find_columns(question, candidates)]


def _detect_aggregation(question: str) -> str | None:
    text = _normalize(question)
    for phrase, aggregation in sorted(_AGGREGATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        if f" {phrase} " in f" {text} ":
            return aggregation
    return None


def _detect_frequency(question: str) -> str | None:
    text = _normalize(question)
    if any(word in text.split() for word in ("hour", "hourly")):
        return "1h"
    if any(word in text.split() for word in ("week", "weekly")):
        return "1w"
    if any(word in text.split() for word in ("month", "monthly")):
        return "1mo"
    if any(word in text.split() for word in ("day", "daily")):
        return "1d"
    return None


def _detect_limit(question: str) -> int | None:
    match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", question.lower())
    return int(match.group(1)) if match else None


def _detect_sort(question: str) -> str | None:
    text = _normalize(question)
    if any(term in text for term in ("highest", "largest", "maximum", "top")):
        return "desc"
    if any(term in text for term in ("lowest", "smallest", "minimum", "bottom")):
        return "asc"
    return None


def plan_query(
    question: str,
    *,
    columns: Sequence[str],
    numeric_columns: Sequence[str] | None = None,
    datetime_columns: Sequence[str] | None = None,
) -> QueryPlan:
    """Convert a supported natural-language question into a structured plan.

    The planner only references columns supplied by the caller. Ambiguous or
    unsupported requests fail explicitly instead of guessing.
    """
    if not question or not question.strip():
        raise PlanningError("Question cannot be empty.")
    if not columns:
        raise PlanningError("At least one dataset column is required.")

    text = _normalize(question)

    if any(term in text for term in ("data quality", "quality check", "quality report", "missing values", "duplicates")):
        return QueryPlan(operation="quality", question=question)

    if any(term in text for term in ("profile dataset", "profile the dataset", "dataset profile", "profile the data")):
        return QueryPlan(operation="profile", question=question)

    mentioned_numeric = _numeric_columns(question, columns, numeric_columns)
    mentioned_datetime = _datetime_columns(question, columns, datetime_columns)

    if "correlation" in text or "correlated" in text:
        selected = mentioned_numeric
        if len(selected) < 2:
            raise PlanningError("Correlation requires at least two explicitly named numeric columns.")
        return QueryPlan(
            operation="correlation",
            columns=selected,
            question=question,
        )

    if any(term in text for term in ("trend", "over time", "time series", "by day", "daily", "by week", "weekly", "by month", "monthly")):
        if not mentioned_datetime:
            raise PlanningError("Time-series analysis requires an explicitly named date/datetime column.")
        metric = mentioned_numeric[0] if mentioned_numeric else None
        aggregation = _detect_aggregation(question) or "count"
        return QueryPlan(
            operation="time_series",
            datetime_column=mentioned_datetime[0],
            metric=metric,
            aggregation=aggregation,
            frequency=_detect_frequency(question) or "1d",
            question=question,
        )

    aggregation = _detect_aggregation(question)
    grouped_intent = any(term in text for term in (" by ", " per ", " each ", " grouped", "group by"))
    ranking_intent = any(term in text for term in ("highest", "largest", "maximum", "lowest", "smallest", "minimum", "top", "bottom"))
    if aggregation and (grouped_intent or ranking_intent):
        # A grouping column may be numeric in the source data (for example,
        # NYC TLC encodes payment_type as an integer). Exclude the selected
        # metric and datetime columns rather than assuming every numeric
        # column is a metric.
        ranking_intent = any(term in text for term in ("highest", "largest", "maximum", "lowest", "smallest", "minimum", "top", "bottom"))
        # In ranking questions such as "highest average fare_amount by payment_type",
        # the metric is usually the numeric column nearest the aggregation phrase;
        # for ordinary "average X by Y" questions, preserve the existing first-match behavior.
        selected_metric = (mentioned_numeric[-1] if ranking_intent else mentioned_numeric[0]) if mentioned_numeric else None
        group_candidates = [
            c for c in columns
            if c != selected_metric and c not in mentioned_datetime
        ]
        group_matches = _find_columns(question, group_candidates)
        if not group_matches:
            raise PlanningError("Grouped analysis requires an explicitly named grouping column.")
        if aggregation != "count" and not selected_metric:
            raise PlanningError("Grouped metric analysis requires an explicitly named numeric metric.")
        return QueryPlan(
            operation="aggregate",
            group_by=[group_matches[0]],
            metric=selected_metric,
            aggregation=aggregation,
            sort_direction=_detect_sort(question),
            limit=_detect_limit(question),
            question=question,
        )

    if any(term in text for term in ("describe", "descriptive statistics", "summary statistics", "statistics for", "distribution of")):
        if not mentioned_numeric:
            raise PlanningError("Descriptive statistics require at least one explicitly named numeric column.")
        return QueryPlan(operation="describe", columns=mentioned_numeric, question=question)

    raise PlanningError(
        "Unsupported analytical request. Supported intents: profile, quality, "
        "describe, grouped aggregation, correlation, and time-series analysis."
    )
