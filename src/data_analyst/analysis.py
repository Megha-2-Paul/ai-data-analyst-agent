"""Reusable analytical operations for structured datasets."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

SUPPORTED_AGGREGATIONS = {"count", "sum", "mean", "median", "min", "max", "std"}


def _require_column(df: pl.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {df.columns}")


def _require_columns(df: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Analysis references missing column(s): {', '.join(missing)}")


def _result(analysis: str, parameters: dict[str, Any], data: Any, **metadata: Any) -> dict[str, Any]:
    return {"analysis": analysis, "parameters": parameters, "result": data, "metadata": metadata}


def _linear_quantile(values: pl.Series, quantile: float) -> float:
    """Calculate a percentile using NumPy's explicit linear interpolation."""
    return float(np.quantile(values.to_numpy(), quantile, method="linear"))


def describe_numeric(df: pl.DataFrame, column: str) -> dict[str, Any]:
    """Calculate basic descriptive statistics for one numeric column."""
    _require_column(df, column)
    series = df.get_column(column)
    if not series.dtype.is_numeric():
        raise TypeError(f"Column '{column}' is not numeric.")
    values = series.drop_nulls()
    if not len(values):
        return {"column": column, "count": 0}
    return {"column": column, "count": len(values), "missing": series.null_count(), "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "q25": _linear_quantile(values, .25), "q75": _linear_quantile(values, .75), "max": values.max()}


def value_counts(df: pl.DataFrame, column: str, limit: int = 10) -> pl.DataFrame:
    """Return the most frequent values in a column."""
    _require_column(df, column)
    return df.get_column(column).value_counts(sort=True).head(limit)


def grouped_summary(df: pl.DataFrame, group_by: str, metric: str, operation: str = "mean") -> pl.DataFrame:
    """Aggregate a numeric metric by one grouping column."""
    _require_column(df, group_by)
    _require_column(df, metric)
    if not df.get_column(metric).dtype.is_numeric():
        raise TypeError(f"Metric '{metric}' is not numeric.")
    if operation not in SUPPORTED_AGGREGATIONS - {"std"}:
        raise ValueError(f"Unsupported operation '{operation}'. Choose from {sorted(SUPPORTED_AGGREGATIONS - {'std'})}.")
    if operation == "count":
        return df.group_by(group_by).len(name=f"{metric}_count").sort(group_by)
    return df.group_by(group_by).agg(getattr(pl.col(metric), operation)().alias(f"{metric}_{operation}")).sort(group_by)


def describe(df: pl.DataFrame, columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return descriptive statistics for selected numeric columns."""
    selected = list(columns) if columns is not None else [n for n, d in zip(df.columns, df.dtypes) if d.is_numeric()]
    _require_columns(df, selected)
    bad = [n for n in selected if not df.schema[n].is_numeric()]
    if bad:
        raise ValueError(f"Descriptive statistics require numeric column(s): {', '.join(bad)}")
    rows = []
    for column in selected:
        values = df.get_column(column).drop_nulls()
        row = {"column": column, "count": len(values), "missing": df.get_column(column).null_count()}
        if len(values):
            row.update({"mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "p25": _linear_quantile(values, .25), "p50": _linear_quantile(values, .50), "p75": _linear_quantile(values, .75), "p95": _linear_quantile(values, .95), "max": values.max(), "unique": values.n_unique()})
        rows.append(row)
    return _result("descriptive_statistics", {"columns": selected}, rows, row_count=df.height)


def aggregate(df: pl.DataFrame, *, group_by: Sequence[str] | None, metric: str, agg: str) -> dict[str, Any]:
    """Aggregate a metric globally or by one or more grouping columns."""
    _require_columns(df, [metric, *(group_by or [])])
    if agg not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation '{agg}'. Choose from: {', '.join(sorted(SUPPORTED_AGGREGATIONS))}")
    expression = pl.col(metric).count() if agg == "count" else getattr(pl.col(metric), agg)()
    result = df.group_by(list(group_by), maintain_order=True).agg(expression.alias(metric)).to_dicts() if group_by else df.select(expression.alias(metric)).to_dicts()
    return _result("grouped_aggregation" if group_by else "aggregation", {"group_by": list(group_by or []), "metric": metric, "aggregation": agg}, result)


def correlation(df: pl.DataFrame, columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return a Pearson correlation matrix for numeric columns."""
    selected = list(columns) if columns is not None else [n for n, d in zip(df.columns, df.dtypes) if d.is_numeric()]
    _require_columns(df, selected)
    bad = [n for n in selected if not df.schema[n].is_numeric()]
    if bad:
        raise ValueError(f"Correlation requires numeric column(s): {', '.join(bad)}")
    return _result("correlation", {"columns": selected}, df.select(selected).corr().to_dicts())


def time_series(df: pl.DataFrame, *, datetime_column: str, frequency: str = "1d", metric: str | None = None, agg: str = "count") -> dict[str, Any]:
    """Aggregate records over a Date/Datetime column or an integer year column."""
    _require_columns(df, [datetime_column] + ([metric] if metric else []))
    if agg not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation '{agg}'. Choose from: {', '.join(sorted(SUPPORTED_AGGREGATIONS))}")

    metric_name = metric or "row_count"
    expression = (
        pl.len().alias(metric_name)
        if agg == "count" and metric is None
        else (pl.col(metric).count().alias(metric_name) if agg == "count" else getattr(pl.col(metric), agg)().alias(metric_name))
    )

    dtype = df.schema[datetime_column]
    if dtype in (pl.Date, pl.Datetime):
        result = df.sort(datetime_column).group_by_dynamic(
            datetime_column, every=frequency
        ).agg(expression).sort(datetime_column).to_dicts()
    elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64) and frequency == "1y":
        result = df.group_by(datetime_column, maintain_order=True).agg(expression).sort(datetime_column).to_dicts()
    else:
        raise ValueError(
            f"Time analysis requires a Date/Datetime column or an integer year column: '{datetime_column}'"
        )

    return _result(
        "time_series",
        {"datetime_column": datetime_column, "frequency": frequency, "metric": metric, "aggregation": agg},
        result,
    )
