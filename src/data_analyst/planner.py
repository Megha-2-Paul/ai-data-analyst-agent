"""Natural-language query planning for the Stage 1 analytical engine.

The planner is deliberately conservative: it converts common analytical requests
into a structured plan, but it does not execute analysis or invent columns.
The future LLM agent can use the same plan schema as a tool-call contract.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

Operation = Literal[
    "profile",
    "quality",
    "describe",
    "aggregate",
    "correlation",
    "time_series",
    "anomaly_detection",
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
    aggregation: str | None = None
    datetime_column: str | None = None
    frequency: str | None = None
    sort_direction: str | None = None
    limit: int | None = None
    question: str = ""
    assumptions: list[str] = field(default_factory=list)
    filter_column: str | None = None
    filter_value: Any | None = None
    percentage_of_total: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["agg"] = self.aggregation
        return result

    @property
    def agg(self) -> str | None:
        """Compatibility alias for callers using the analysis API's ``agg`` name."""
        return self.aggregation


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
        "years": "1y",
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


def _detect_explicit_filter(
    question: str,
    columns: Sequence[str],
    categorical_values: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[str, Any] | tuple[None, None]:
    """Resolve explicit filters and common value-first phrasing against observed data.

    The value vocabulary is supplied by the caller from the current dataset; the
    planner never assumes domain-specific columns or values.
    """
    for column in sorted(columns, key=len, reverse=True):
        pattern = (
            r"\b(?:where|filter(?:ed)? by)\s+"
            + re.escape(column)
            + r"\s*(?:is|equals|equal to|=)\s*"
            + r"""(?:['"]([^'"]+)['"]|([^,;?.]+))"""
        )
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            value = (match.group(1) or match.group(2) or "").strip()
            if value:
                return column, value

    if not categorical_values:
        return None, None

    normalized_question = f" {_normalize(question)} "
    candidates: list[tuple[str, Any]] = []
    for column, values in categorical_values.items():
        if column not in columns:
            continue
        normalized_column = _normalize(column)
        column_mentioned = bool(
            normalized_column
            and re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_column)}(?![a-z0-9])",
                normalized_question,
            )
        )
        for value in values:
            if value is None:
                continue
            normalized_value = _normalize(str(value))
            if not normalized_value:
                continue
            value_pattern = rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])"
            if not re.search(value_pattern, normalized_question):
                continue
            value_context = bool(
                re.search(
                    rf"\b(?:for|in|within|from|under|where)\s+(?:the\s+)?{re.escape(normalized_value)}\b",
                    normalized_question,
                )
                or re.search(
                    rf"\b{re.escape(normalized_value)}\s+{re.escape(normalized_column)}\b",
                    normalized_question,
                )
            )
            if column_mentioned or value_context:
                candidates.append((column, value))

    unique_candidates: list[tuple[str, Any]] = []
    for candidate in candidates:
        if not any(
            candidate[0] == existing[0]
            and str(candidate[1]).casefold() == str(existing[1]).casefold()
            for existing in unique_candidates
        ):
            unique_candidates.append(candidate)

    if len(unique_candidates) > 1:
        details = ", ".join(f"{column}={value!r}" for column, value in unique_candidates)
        raise PlanningError(
            f"Ambiguous filter value in question; it matches multiple dataset values: {details}."
        )
    return unique_candidates[0] if unique_candidates else (None, None)

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
    categorical_values: Mapping[str, Sequence[Any]] | None = None,
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
    filter_column, filter_value = _detect_explicit_filter(
        question, columns, categorical_values=categorical_values
    )

    anomaly_intent = any(term in text for term in ("anomaly", "anomalies", "outlier", "outliers", "unusual records", "unusual observations"))
    if anomaly_intent:
        selected = mentioned_numeric or list(numeric_columns or [])
        if not selected:
            raise PlanningError("Anomaly detection requires at least one numeric column in the dataset.")
        return QueryPlan(
            operation="anomaly_detection",
            columns=selected,
            question=question,
            assumptions=["Isolation Forest flags statistically unusual rows; it does not determine whether a row is erroneous or fraudulent."],
        )

    if "correlation" in text or "correlated" in text:
        selected = mentioned_numeric
        if len(selected) < 2:
            raise PlanningError("Correlation requires at least two explicitly named numeric columns.")
        return QueryPlan(
            operation="correlation",
            columns=selected,
            question=question,
            filter_column=filter_column,
            filter_value=filter_value,
        )

    frequency = _detect_frequency(question, columns)
    time_series_intent = frequency is not None or any(
        term in text
        for term in ("trend", "over time", "time series", "over the years", "over years")
    )
    if time_series_intent:
        datetime_column = _one_column(mentioned_datetime, "date/datetime column")
        if datetime_column is None:
            year_columns = [column for column in columns if _normalize(column) == "year"]
            if len(year_columns) > 1:
                raise PlanningError("Ambiguous year column; provide an explicit date/datetime column.")
            if len(year_columns) == 1 and any(
                term in text for term in ("year", "years", "yearly", "annual", "annually", "over time")
            ):
                datetime_column = year_columns[0]
        if datetime_column is None:
            raise PlanningError(
                "Time-series analysis requires an explicitly named date/datetime column or year column."
            )

        metric_candidates = [
            column for column in mentioned_numeric if column != datetime_column
        ]
        metric = _one_column(metric_candidates, "numeric metric")
        aggregation = _detect_aggregation(question)
        assumptions = []
        if aggregation is None:
            aggregation = "mean" if metric is not None else "count"
            default_label = "mean" if metric is not None else "count"
            assumptions.append(f"No aggregation was specified; defaulted to {default_label}.")

        if frequency is None:
            frequency = "1y" if _normalize(datetime_column) == "year" else "1d"
            label = "yearly (1y)" if frequency == "1y" else "daily (1d)"
            assumptions.append(f"No time frequency was specified; defaulted to {label}.")

        numeric_set = set(numeric_columns or [])
        datetime_set = set(datetime_columns or [])
        non_temporal_categorical = [
            column for column in columns
            if column not in numeric_set
            and column not in datetime_set
            and column != datetime_column
        ]
        country_columns = [
            column for column in non_temporal_categorical
            if _normalize(column) in ("country", "country name")
        ]
        if country_columns:
            country_match = re.search(
                r"\b([A-Za-z][A-Za-z ._-]{0,60}?)'s\s+",
                question,
                flags=re.IGNORECASE,
            )
            if country_match:
                candidate = country_match.group(1).strip()
                candidate = re.sub(
                    r"^(?:(?:how|what|when|where|why|did|does|has|have|is|was|were|show|give|tell|calculate|compute|find|display)\s+)+",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                )
                candidate = re.sub(
                    r"^(?:the\s+)?(?:sum|average|mean|total|median|minimum|maximum)\s+(?:of\s+)?",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()
                if candidate:
                    filter_column = country_columns[0]
                    filter_value = candidate

        return QueryPlan(
            operation="time_series",
            datetime_column=datetime_column,
            metric=metric,
            aggregation=aggregation,
            frequency=frequency,
            question=question,
            assumptions=assumptions,
            filter_column=filter_column,
            filter_value=filter_value,
        )

    aggregation = _detect_aggregation(question)
    percentage_of_total = any(term in text for term in ("percentage", "percent", "share of total", "proportion of total"))
    grouped_intent = any(
        term in text for term in (" by ", " per ", " each ", " grouped", "group by")
    )
    ranking_intent = any(
        term in text
        for term in ("highest", "largest", "maximum", "lowest", "smallest", "minimum", "top", "bottom")
    )
    if aggregation and (grouped_intent or ranking_intent):
        # Exclude temporal columns from grouping candidates, but allow numeric
        # grouping dimensions such as payment_type. Select the metric only after
        # resolving the grouping column so multiple metrics cannot be guessed.
        # Resolve the grouping dimension before selecting a metric. In ranking
        # questions, the group often appears in a "which <column> has ..." clause
        # rather than a "by/per" clause, and that column may itself be numeric.
        group_candidates = [
            column for column in columns if column not in mentioned_datetime
        ]

        group_matches = _group_columns(question, group_candidates) if grouped_intent else []
        if not group_matches:
            which_match = re.search(
                r"\bwhich\s+(.+?)\s+(?:has|have|is|was|were)\b",
                question,
                flags=re.IGNORECASE,
            )
            if which_match:
                group_matches = _find_columns(which_match.group(1), group_candidates)
        if not group_matches:
            grouping_pattern = re.compile(
                r"\b(?:by|per|each)\s+(.+?)(?=\s+(?:and|with|over|for)\b|[?,.]|$)",
                re.IGNORECASE,
            )
            for match in grouping_pattern.finditer(question):
                group_matches.extend(_find_columns(match.group(1), group_candidates))
        if not group_matches:
            group_matches = _find_columns(question, group_candidates)

        group_matches = list(dict.fromkeys(group_matches))
        if not group_matches:
            raise PlanningError("Grouped analysis requires an explicitly named grouping column.")

        metric_candidates = [
            column for column in mentioned_numeric
            if column not in group_matches and column not in mentioned_datetime
        ]
        if aggregation != "count":
            selected_metric = _one_column(metric_candidates, "numeric metric")
            if selected_metric is None:
                raise PlanningError("Grouped metric analysis requires an explicitly named numeric metric.")
        else:
            selected_metric = metric_candidates[0] if metric_candidates else None

        return QueryPlan(
            operation="aggregate",
            group_by=group_matches,
            metric=selected_metric,
            aggregation=aggregation,
            filter_column=filter_column,
            filter_value=filter_value,
            percentage_of_total=percentage_of_total,
            sort_direction=_detect_sort(question),
            limit=(
                _detect_limit(question)
                if _detect_limit(question) is not None
                else 1 if ranking_intent and re.search(r"\bwhich\b", text)
                else None
            ),
            question=question,
        )

    if aggregation and not grouped_intent and not ranking_intent:
        selected_metric = _one_column(mentioned_numeric, "numeric metric")
        if aggregation != "count" and selected_metric is None:
            raise PlanningError("A global aggregation requires an explicitly named numeric metric.")
        return QueryPlan(
            operation="aggregate",
            metric=selected_metric,
            aggregation=aggregation,
            question=question,
            filter_column=filter_column,
            filter_value=filter_value,
        )

    if any(term in text for term in ("describe", "descriptive statistics", "summary statistics", "statistics for", "distribution of")):
        if not mentioned_numeric:
            raise PlanningError("Descriptive statistics require at least one explicitly named numeric column.")
        return QueryPlan(operation="describe", columns=mentioned_numeric, question=question, filter_column=filter_column, filter_value=filter_value)

    raise PlanningError(
        "Unsupported analytical request. Supported intents: profile, quality, "
        "describe, grouped aggregation, correlation, time-series analysis, and anomaly detection."
    )
