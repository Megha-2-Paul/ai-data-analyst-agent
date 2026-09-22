"""Safe, dataset-agnostic cleaning plans and transformations.

Stage 4 deliberately separates safe normalization from statistical/domain-dependent
cleaning. No imputation or outlier removal is performed automatically.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import polars as pl


SAFE_OPERATIONS = {
    "normalize_column_names",
    "trim_strings",
    "empty_strings_to_null",
    "parse_datetime",
    "drop_empty_columns",
}


class CleaningError(ValueError):
    """Raised when a cleaning plan is invalid or cannot be applied."""


@dataclass(frozen=True)
class CleaningStep:
    """One explicit, traceable cleaning operation."""

    step_id: str
    operation: str
    columns: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    automatic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleaningPlan:
    """Validated sequence of safe transformations plus recommendations."""

    steps: list[CleaningStep]
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class CleaningResult:
    """Result of applying a cleaning plan without overwriting the source data."""

    dataframe: pl.DataFrame
    applied_steps: list[dict[str, Any]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied_steps": list(self.applied_steps),
            "recommendations": list(self.recommendations),
            "row_count": self.dataframe.height,
            "column_count": self.dataframe.width,
            "columns": list(self.dataframe.columns),
        }


def _normalized_name(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip().lower()).strip("_")
    return value or "column"


def _unique_names(names: Sequence[str]) -> list[str]:
    used: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        base = _normalized_name(name)
        count = used.get(base, 0)
        candidate = base if count == 0 else f"{base}_{count + 1}"
        while candidate in result:
            count += 1
            candidate = f"{base}_{count + 1}"
        used[base] = count + 1
        result.append(candidate)
    return result


def _looks_temporal(name: str) -> bool:
    normalized = _normalized_name(name)
    return any(token in normalized.split("_") for token in ("date", "datetime", "timestamp", "time"))


def _datetime_candidate(series: pl.Series) -> tuple[pl.Series, float] | None:
    non_null = series.drop_nulls()
    if len(non_null) < 2:
        return None
    parsed = non_null.cast(pl.String).str.to_datetime(strict=False)
    success = parsed.is_not_null().sum() / len(non_null)
    return parsed, float(success)


def build_cleaning_plan(df: pl.DataFrame) -> CleaningPlan:
    """Infer only transformations that are safe to apply without domain rules."""

    steps: list[CleaningStep] = []
    recommendations: list[str] = []

    normalized = _unique_names(df.columns)
    if normalized != df.columns:
        steps.append(
            CleaningStep(
                "step_1",
                "normalize_column_names",
                reason="Make column names stable and analysis-friendly without changing values.",
            )
        )

    next_id = len(steps) + 1
    string_columns = [name for name, dtype in df.schema.items() if dtype == pl.String]
    if string_columns:
        steps.append(
            CleaningStep(
                f"step_{next_id}",
                "trim_strings",
                columns=string_columns,
                reason="Remove accidental leading/trailing whitespace from string values.",
            )
        )
        next_id += 1
        steps.append(
            CleaningStep(
                f"step_{next_id}",
                "empty_strings_to_null",
                columns=string_columns,
                reason="Represent blank string values consistently as missing values.",
            )
        )
        next_id += 1

    temporal_candidates: list[str] = []
    for name in string_columns:
        if not _looks_temporal(name):
            continue
        candidate = _datetime_candidate(df.get_column(name))
        if candidate and candidate[1] >= 0.95:
            temporal_candidates.append(name)

    if temporal_candidates:
        steps.append(
            CleaningStep(
                f"step_{next_id}",
                "parse_datetime",
                columns=temporal_candidates,
                parameters={"success_threshold": 0.95},
                reason="Column name indicates temporal data and at least 95% of non-null values parse as datetimes.",
            )
        )
        next_id += 1

    empty_columns = [
        name for name in df.columns
        if df.get_column(name).null_count() == df.height
    ]
    if empty_columns:
        steps.append(
            CleaningStep(
                f"step_{next_id}",
                "drop_empty_columns",
                columns=empty_columns,
                reason="Remove columns containing no observed values.",
            )
        )

    null_columns = [
        name for name in df.columns
        if df.get_column(name).null_count() > 0
    ]
    if null_columns:
        recommendations.append(
            "Missing values remain in the data. Review their cause before choosing imputation or row deletion."
        )

    duplicate_count = df.height - df.unique().height
    if duplicate_count:
        recommendations.append(
            f"{duplicate_count} exact duplicate row(s) detected. Removal is not automatic because duplicates may be legitimate observations."
        )

    numeric_columns = [
        name for name, dtype in df.schema.items() if dtype.is_numeric()
    ]
    if numeric_columns:
        outlier_columns = []
        for name in numeric_columns:
            values = df.get_column(name).drop_nulls()
            if len(values) < 4:
                continue
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            if q1 is not None and q3 is not None and q3 != q1:
                lower = q1 - 1.5 * (q3 - q1)
                upper = q3 + 1.5 * (q3 - q1)
                if values.filter((values < lower) | (values > upper)).len():
                    outlier_columns.append(name)
        if outlier_columns:
            recommendations.append(
                "Potential IQR outliers detected in: "
                + ", ".join(outlier_columns)
                + ". Outliers are not removed automatically."
            )

    return CleaningPlan(steps=steps, recommendations=recommendations)


def validate_cleaning_plan(plan: CleaningPlan, columns: Sequence[str]) -> CleaningPlan:
    """Validate a cleaning plan before any transformation is applied."""

    known = set(columns)
    seen_ids: set[str] = set()
    for step in plan.steps:
        if step.step_id in seen_ids:
            raise CleaningError(f"Duplicate cleaning step ID: {step.step_id!r}")
        seen_ids.add(step.step_id)
        if step.operation not in SAFE_OPERATIONS:
            raise CleaningError(
                f"Unsupported automatic cleaning operation: {step.operation!r}"
            )
        unknown = set(step.columns) - known
        if unknown and step.operation != "normalize_column_names":
            raise CleaningError(
                f"Cleaning step {step.step_id!r} references unknown columns: {sorted(unknown)}"
            )
    return plan


def apply_cleaning_plan(df: pl.DataFrame, plan: CleaningPlan) -> CleaningResult:
    """Apply validated safe transformations to a copy of the dataframe."""

    validate_cleaning_plan(plan, df.columns)
    working = df.clone()
    applied: list[dict[str, Any]] = []

    for step in plan.steps:
        before_shape = (working.height, working.width)

        if step.operation == "normalize_column_names":
            working = working.rename(dict(zip(working.columns, _unique_names(working.columns))))
        elif step.operation == "trim_strings":
            if step.columns:
                working = working.with_columns(
                    pl.col(step.columns).cast(pl.String).str.strip_chars().alias(name)
                    for name in step.columns
                )
        elif step.operation == "empty_strings_to_null":
            if step.columns:
                working = working.with_columns(
                    pl.when(pl.col(name).str.strip_chars() == "")
                    .then(pl.lit(None, dtype=pl.String))
                    .otherwise(pl.col(name))
                    .alias(name)
                    for name in step.columns
                )
        elif step.operation == "parse_datetime":
            for name in step.columns:
                if name not in working.columns:
                    continue
                working = working.with_columns(
                    pl.col(name).cast(pl.String).str.to_datetime(strict=False).alias(name)
                )
        elif step.operation == "drop_empty_columns":
            working = working.drop([name for name in step.columns if name in working.columns])

        applied.append(
            {
                "step_id": step.step_id,
                "operation": step.operation,
                "columns": list(step.columns),
                "before_shape": list(before_shape),
                "after_shape": [working.height, working.width],
            }
        )

    return CleaningResult(working, applied, list(plan.recommendations))
