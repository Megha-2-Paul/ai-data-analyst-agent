"""DuckDB execution layer for analytical SQL."""

from pathlib import Path
from typing import Any

import duckdb


def query_parquet(path: str | Path, sql: str) -> list[dict[str, Any]]:
    """Execute a read-only SQL query against a Parquet file.

    The file is exposed to SQL as the table name ``dataset``. The SQL must be
    a single SELECT/WITH statement in Stage 1; write operations are rejected.
    """
    statement = sql.strip().rstrip(";").lstrip().lower()
    if not (statement.startswith("select ") or statement.startswith("with ")):
        raise ValueError("Only read-only SELECT/WITH queries are allowed in Stage 1.")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    escaped = str(file_path).replace("'", "''")
    wrapped = sql.replace("{dataset}", f"read_parquet('{escaped}')")
    with duckdb.connect(database=":memory:", read_only=False) as con:
        return con.execute(wrapped).fetchdf().to_dict(orient="records")
