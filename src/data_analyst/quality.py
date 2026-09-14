"""Dataset quality checks and configurable validation rules."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

import polars as pl


Severity = str


def quality_report(
    df: pl.DataFrame,
    *,
    non_negative_columns: Iterable[str] | None = None,
    datetime_order_pairs: Iterable[tuple[str, str]] | None = None,
    expected_categories: Mapping[str, Iterable[Any]] | None = None,
    maximum_values: Mapping[str, float] | None = None,
    date_ranges: Mapping[str, tuple[datetime, datetime]] | None = None,
    example_limit: int = 5,
) -> dict[str, Any]:
    """Return generic quality findings for a dataframe.

    Statistical checks are always available. Domain-validity checks are
    configurable so the engine remains dataset-agnostic instead of embedding
    rules for a particular dataset.
    """
    nulls = {
        name: df.get_column(name).null_count()
        for name in df.columns
        if df.get_column(name).null_count() > 0
    }
    duplicate_rows = df.height - df.unique().height

    numeric_outliers: dict[str, int] = {}
    for name, dtype in zip(df.columns, df.dtypes):
        if not dtype.is_numeric():
            continue
        values = df.get_column(name).drop_nulls()
        if len(values) < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(values.filter((values < lower) | (values > upper)).len())
        if count:
            numeric_outliers[name] = count

    findings = _validation_findings(
        df,
        non_negative_columns=non_negative_columns or (),
        datetime_order_pairs=datetime_order_pairs or (),
        expected_categories=expected_categories or {},
        maximum_values=maximum_values or {},
        date_ranges=date_ranges or {},
        example_limit=example_limit,
    )

    total_cells = df.height * df.width
    null_cells = sum(nulls.values())
    completeness = 100.0 if total_cells == 0 else (1 - null_cells / total_cells) * 100

    return {
        "row_count": df.height,
        "column_count": df.width,
        "duplicate_row_count": duplicate_rows,
        "null_counts": nulls,
        "outlier_counts_iqr": numeric_outliers,
        "completeness_pct": round(completeness, 2),
        "findings": findings,
        "warnings": _warnings(nulls, duplicate_rows, numeric_outliers, findings),
    }


def _validation_findings(
    df: pl.DataFrame,
    *,
    non_negative_columns: Iterable[str],
    datetime_order_pairs: Iterable[tuple[str, str]],
    expected_categories: Mapping[str, Iterable[Any]],
    maximum_values: Mapping[str, float],
    date_ranges: Mapping[str, tuple[datetime, datetime]],
    example_limit: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    for column in non_negative_columns:
        _require_column(df, column)
        values = df.get_column(column).drop_nulls()
        invalid = values.filter(values < 0)
        if len(invalid):
            findings.append(
                _finding(
                    "NON_NEGATIVE",
                    "ERROR",
                    column,
                    len(invalid),
                    df.height,
                    f"Negative values found in '{column}', which is configured as non-negative.",
                    examples=invalid.head(example_limit).to_list(),
                )
            )

    for start_column, end_column in datetime_order_pairs:
        _require_column(df, start_column)
        _require_column(df, end_column)
        invalid = df.filter(df.get_column(start_column) > df.get_column(end_column))
        if invalid.height:
            findings.append(
                _finding(
                    "DATETIME_ORDER",
                    "ERROR",
                    f"{start_column} <= {end_column}",
                    invalid.height,
                    df.height,
                    f"'{start_column}' occurs after '{end_column}' in some rows.",
                )
            )

    for column, allowed_values in expected_categories.items():
        _require_column(df, column)
        allowed = set(allowed_values)
        invalid = df.filter(~pl.col(column).is_null() & ~pl.col(column).is_in(allowed))
        if invalid.height:
            examples = invalid.get_column(column).unique().head(example_limit).to_list()
            findings.append(
                _finding(
                    "EXPECTED_CATEGORY",
                    "WARNING",
                    column,
                    invalid.height,
                    df.height,
                    f"Unexpected categorical values found in '{column}'.",
                    examples=examples,
                )
            )

    for column, maximum in maximum_values.items():
        _require_column(df, column)
        invalid = df.filter(pl.col(column).is_not_null() & (pl.col(column) > maximum))
        if invalid.height:
            findings.append(
                _finding(
                    "MAXIMUM_VALUE",
                    "WARNING",
                    column,
                    invalid.height,
                    df.height,
                    f"Values in '{column}' exceed the configured maximum of {maximum}.",
                    examples=invalid.get_column(column).head(example_limit).to_list(),
                )
            )

    for column, (minimum, maximum) in date_ranges.items():
        _require_column(df, column)
        invalid = df.filter(
            pl.col(column).is_not_null()
            & ((pl.col(column) < minimum) | (pl.col(column) > maximum))
        )
        if invalid.height:
            findings.append(
                _finding(
                    "DATE_RANGE",
                    "WARNING",
                    column,
                    invalid.height,
                    df.height,
                    f"Dates in '{column}' fall outside the configured range.",
                    examples=[str(value) for value in invalid.get_column(column).head(example_limit).to_list()],
                )
            )

    return findings


def _finding(
    check: str,
    severity: Severity,
    target: str,
    affected_rows: int,
    total_rows: int,
    message: str,
    *,
    examples: list[Any] | None = None,
) -> dict[str, Any]:
    affected_pct = 0.0 if total_rows == 0 else round((affected_rows / total_rows) * 100, 4)
    result: dict[str, Any] = {
        "check": check,
        "severity": severity,
        "target": target,
        "affected_rows": affected_rows,
        "affected_pct": affected_pct,
        "message": message,
    }
    if examples:
        result["examples"] = examples
    return result


def _require_column(df: pl.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Validation rule references missing column: '{column}'")


def _warnings(
    nulls: dict[str, int],
    duplicates: int,
    outliers: dict[str, int],
    findings: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if nulls:
        warnings.append(f"Missing values detected in {len(nulls)} column(s).")
    if duplicates:
        warnings.append(f"{duplicates} duplicate row(s) detected.")
    if outliers:
        warnings.append(f"Potential IQR outliers detected in {len(outliers)} numeric column(s).")
    for finding in findings:
        warnings.append(f"{finding['severity']}: {finding['message']}")
    return warnings
