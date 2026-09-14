"""Dataset ingestion helpers."""

from pathlib import Path
from typing import Union

import polars as pl

PathLike = Union[str, Path]
SUPPORTED_FORMATS = {".csv", ".parquet"}


def load_dataset(path: PathLike) -> pl.DataFrame:
    """Load a CSV or Parquet dataset into a Polars DataFrame."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    if suffix == ".csv":
        return pl.read_csv(file_path, try_parse_dates=True)
    return pl.read_parquet(file_path)


def scan_dataset(path: PathLike) -> pl.LazyFrame:
    """Create a lazy scan for a CSV or Parquet dataset."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(file_path)
    if suffix == ".csv":
        return pl.scan_csv(file_path, try_parse_dates=True)
    raise ValueError(
        f"Unsupported format '{suffix}'. Supported formats: {sorted(SUPPORTED_FORMATS)}"
    )
