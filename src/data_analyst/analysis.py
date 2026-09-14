"""Reusable analytical operations for structured datasets."""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

SUPPORTED_AGGREGATIONS = {"count", "sum", "mean", "median", "min", "max", "std"}


def _require_columns(df: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Analysis references missing column(s): {', '.join(missing)}")


def _result(analysis: str, parameters: dict[str, Any], data: Any, **metadata: Any) -> dict[str, Any]:
    return {"analysis": analysis, "parameters": parameters, "result": data, "metadata": metadata}


def describe(df: pl.DataFrame, columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return descriptive statistics for numeric columns."""
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
            row.update({"mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "p25": values.quantile(.25), "p50": values.quantile(.50), "p75": values.quantile(.75), "p95": values.quantile(.95), "max": values.max(), "unique": values.n_unique()})
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
    """Aggregate records over a Date/Datetime column."""
    _require_columns(df, [datetime_column] + ([metric] if metric else []))
    if df.schema[datetime_column] not in (pl.Date, pl.Datetime):
        raise ValueError(f"Time analysis requires a Date or Datetime column: '{datetime_column}'")
    if agg not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation '{agg}'. Choose from: {', '.join(sorted(SUPPORTED_AGGREGATIONS))}")
    metric_name = metric or "row_count"
    expression = pl.len().alias(metric_name) if agg == "count" and metric is None else (pl.col(metric).count().alias(metric_name) if agg == "count" else getattr(pl.col(metric), agg)().alias(metric_name))
    result = df.sort(datetime_column).group_by_dynamic(datetime_column, every=frequency).agg(expression).sort(datetime_column).to_dicts()
    return _result("time_series", {"datetime_column": datetime_column, "frequency": frequency, "metric": metric, "aggregation": agg}, result)
