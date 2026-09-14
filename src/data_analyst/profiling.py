"""Dataset profiling without dataset-specific assumptions."""

from datetime import datetime
from typing import Any

import polars as pl


def profile_dataset(df: pl.DataFrame) -> dict[str, Any]:
    """Return a JSON-friendly profile of the supplied dataframe."""
    columns: list[dict[str, Any]] = []

    for name, dtype in zip(df.columns, df.dtypes):
        series = df.get_column(name)
        item: dict[str, Any] = {
            "name": name,
            "dtype": str(dtype),
            "null_count": series.null_count(),
            "null_pct": round(series.null_count() / len(df) * 100, 4) if len(df) else 0.0,
            "unique_count": series.n_unique(),
        }

        if dtype.is_numeric():
            values = series.drop_nulls()
            if len(values):
                item.update(
                    {
                        "min": _json_value(values.min()),
                        "max": _json_value(values.max()),
                        "mean": _json_value(values.mean()),
                        "median": _json_value(values.median()),
                    }
                )
        elif dtype in (pl.Date, pl.Datetime):
            values = series.drop_nulls()
            if len(values):
                item.update(
                    {
                        "min": _json_value(values.min()),
                        "max": _json_value(values.max()),
                    }
                )
        else:
            item["top_values"] = {
                str(k): int(v)
                for k, v in series.value_counts(sort=True).head(5).iter_rows()
            }

        columns.append(item)

    return {
        "row_count": df.height,
        "column_count": df.width,
        "columns": columns,
        "duplicate_row_count": df.height - df.unique().height,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value
