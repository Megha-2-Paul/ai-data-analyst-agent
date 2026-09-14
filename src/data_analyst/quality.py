"""Dataset quality checks."""

from typing import Any

import polars as pl


def quality_report(df: pl.DataFrame) -> dict[str, Any]:
    """Return quality findings for a dataframe.

    Checks are intentionally generic. Domain-specific validity rules will be
    added later as configurable validation rules rather than hard-coded into
    the core engine.
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
        "warnings": _warnings(nulls, duplicate_rows, numeric_outliers),
    }


def _warnings(
    nulls: dict[str, int], duplicates: int, outliers: dict[str, int]
) -> list[str]:
    warnings: list[str] = []
    if nulls:
        warnings.append(f"Missing values detected in {len(nulls)} column(s).")
    if duplicates:
        warnings.append(f"{duplicates} duplicate row(s) detected.")
    if outliers:
        warnings.append(f"Potential IQR outliers detected in {len(outliers)} numeric column(s).")
    return warnings
