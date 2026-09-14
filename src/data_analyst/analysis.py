"""Generic descriptive and grouped analysis operations."""

from typing import Any

import polars as pl


def describe_numeric(df: pl.DataFrame, column: str) -> dict[str, Any]:
    """Calculate basic descriptive statistics for a numeric column."""
    _require_column(df, column)
    series = df.get_column(column)
    if not series.dtype.is_numeric():
        raise TypeError(f"Column '{column}' is not numeric.")

    values = series.drop_nulls()
    if not len(values):
        return {"column": column, "count": 0}

    return {
        "column": column,
        "count": len(values),
        "missing": series.null_count(),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "min": values.min(),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "max": values.max(),
    }


def value_counts(df: pl.DataFrame, column: str, limit: int = 10) -> pl.DataFrame:
    """Return the most frequent values in a column."""
    _require_column(df, column)
    return df.get_column(column).value_counts(sort=True).head(limit)


def grouped_summary(
    df: pl.DataFrame, group_by: str, metric: str, operation: str = "mean"
) -> pl.DataFrame:
    """Aggregate a numeric metric by a categorical/grouping column."""
    _require_column(df, group_by)
    _require_column(df, metric)
    if not df.get_column(metric).dtype.is_numeric():
        raise TypeError(f"Metric '{metric}' is not numeric.")

    allowed = {"mean", "sum", "min", "max", "median", "count"}
    if operation not in allowed:
        raise ValueError(f"Unsupported operation '{operation}'. Choose from {sorted(allowed)}.")

    if operation == "count":
        return df.group_by(group_by).len(name=f"{metric}_count").sort(group_by)

    return (
        df.group_by(group_by)
        .agg(getattr(pl.col(metric), operation)().alias(f"{metric}_{operation}"))
        .sort(group_by)
    )


def _require_column(df: pl.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {df.columns}")
