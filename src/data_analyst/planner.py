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
    """Structured description of one analytical request; it is not executed."""

    operation: Operation
    columns: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    metric: str | None = None
    agg: str | None = None
    datetime_column: str | None = None
    frequency: str | None = None
    sort_direction: str | None = None
    limit: int | None = None
    question: str = ""
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["aggregation"] = self.agg
        return result

    @property
    def aggregation(self) -> str | None:
        """Compatibility alias for the analysis API's ``agg`` parameter."""
        return self.agg


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


def _column_matches(question: str, columns: Sequence[str]) -> list[tuple[int, int, str]]:
    normalized_question = _normalize(question)
    matches: list[tuple[int, int, str]] = []
    for column in columns:
        candidate = _normalize(column)
        if not candidate:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])"
        matches.extend(
            (match.start(), match.end(), column)
            for match in re.finditer(pattern, normalized_question)
        )

    columns_by_span: dict[tuple[int, int], list[str]] = {}
    for start, end, column in matches:
        span_columns = columns_by_span.setdefault((start, end), [])
        if column not in span_columns:
            span_columns.append(column)
    for (start, end), span_columns in columns_by_span.items():
        if len(span_columns) > 1:
            match = normalized_question[start:end]
            raise PlanningError(
                f"Ambiguous column reference '{match}' matches multiple schema columns: "
                f"{', '.join(span_columns)}."
            )

    selected: list[tuple[int, int, str]] = []
    for start, end, column in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < selected_end and end > selected_start for selected_start, selected_end, _ in selected):
            continue
        selected.append((start, end, column))
    return selected


def _find_columns(question: str, columns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(column for _, _, column in _column_matches(question, columns)))


def _numeric_columns(question: str, columns: Sequence[str], numeric_columns: Sequence[str] | None) -> list[str]:
    candidates = columns if numeric_columns is None else numeric_columns
    return _find_columns(question, candidates)


def _datetime_columns(question: str, columns: Sequence[str], datetime_columns: Sequence[str] | None) -> list[str]:
    candidates = columns if datetime_columns is None else datetime_columns
    return _find_columns(question, candidates)


def _detect_aggregation(question: str) -> str | None:
    text = _normalize(question)
    matches = {
        aggregation
        for phrase, aggregation in _AGGREGATIONS.items()
        if f" {phrase} " in f" {text} "
    }
    if len(matches) > 1:
        raise PlanningError("Conflicting aggregation operations were requested.")
    return next(iter(matches), None)


def _detect_frequency(question: str, columns: Sequence[str]) -> str | None:
    text = _normalize(question)
    column_spans = [(start, end) for start, end, _ in _column_matches(question, columns)]
    units = {
        "year": "1y",
        "yearly": "1y",
        "annual": "1y",
        "annually": "1y",
        "hour": "1h",
        "hourly": "1h",
        "week": "1w",
        "weekly": "1w",
        "month": "1mo",
        "monthly": "1mo",
        "day": "1d",
        "daily": "1d",
    }
    matches = set()
    for word, frequency in units.items():
        for match in re.finditer(rf"\b{re.escape(word)}\b", text):
            if not any(start < match.end() and end > match.start() for start, end in column_spans):
                matches.add(frequency)

    unsupported = []
    for word in ("quarter", "quarterly"):
        for match in re.finditer(rf"\b{word}\b", text):
            if not any(start < match.end() and end > match.start() for start, end in column_spans):
                unsupported.append(word)
    if unsupported:
        requested = unsupported[0]
        raise PlanningError(
            f"Unsupported time frequency '{requested}'. Supported frequencies are hourly, daily, weekly, monthly, and yearly."
        )
    if len(matches) > 1:
        raise PlanningError("Conflicting time frequencies were requested.")
    return next(iter(matches), None)


def _detect_limit(question: str) -> int | None:
    match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", question.lower())
    return int(match.group(1)) if match else None


def _detect_sort(question: str) -> str | None:
    text = _normalize(question)
    directions = set()
    if any(term in text.split() for term in ("highest", "largest", "maximum", "top")):
        directions.add("desc")
    if any(term in text.split() for term in ("lowest", "smallest", "minimum", "bottom")):
        directions.add("asc")
    if len(directions) > 1:
        raise PlanningError("Conflicting sort directions were requested.")
    return next(iter(directions), None)


def _one_column(columns: Sequence[str], purpose: str) -> str | None:
    if len(columns) > 1:
        raise PlanningError(f"Name exactly one {purpose}; found: {', '.join(columns)}.")
    return columns[0] if columns else None


def _group_columns(question: str, columns: Sequence[str]) -> list[str]:
    text = _normalize(question)
    top_groups = re.search(r"\b(?:top|bottom)\s+\d+\s+(.+?)\s+groups?\s+by\b", text)
    if top_groups:
        group_clause = top_groups.group(1)
    else:
        marker = re.search(r"\b(?:group\s+by|grouped\s+by|by|per|each)\b", text)
        if marker is None:
            return []
        group_clause = text[marker.end():]
    return _find_columns(group_clause, columns)


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

    if text == "profile" or any(term in text for term in ("profile dataset", "profile the dataset", "dataset profile", "profile the data")):
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

    frequency = _detect_frequency(question, columns)
    if frequency is not None or any(term in text for term in ("trend", "over time", "time series")):
        datetime_column = _one_column(mentioned_datetime, "date/datetime column")
        if datetime_column is None:
            raise PlanningError("Time-series analysis requires an explicitly named date/datetime column.")
        metric = _one_column(mentioned_numeric, "numeric metric")
        aggregation = _detect_aggregation(question)
        assumptions = []
        if aggregation is None:
            aggregation = "count"
            assumptions.append("No aggregation was specified; defaulted to count.")
        if frequency is None:
            frequency = "1d"
            assumptions.append("No time frequency was specified; defaulted to daily (1d).")
        return QueryPlan(
            operation="time_series",
            datetime_column=datetime_column,
            metric=metric,
            agg=aggregation,
            frequency=frequency,
            question=question,
            assumptions=assumptions,
        )

    aggregation = _detect_aggregation(question)
    if aggregation and any(term in text for term in (" by ", " per ", " each ", " grouped", "group by")):
        group_matches = _group_columns(question, columns)
        if not group_matches:
            raise PlanningError("Grouped analysis requires an explicitly named grouping column.")
        metric_candidates = [column for column in mentioned_numeric if column not in group_matches]
        metric = _one_column(metric_candidates, "numeric metric")
        if aggregation != "count" and metric is None:
            raise PlanningError("Grouped metric analysis requires an explicitly named numeric metric.")
        return QueryPlan(
            operation="aggregate",
            group_by=group_matches,
            metric=metric,
            agg=aggregation,
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
