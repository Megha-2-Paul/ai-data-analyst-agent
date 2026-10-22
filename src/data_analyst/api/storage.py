"""Controlled local storage for uploaded source files."""

import os
import uuid
from pathlib import Path

from data_analyst.ingestion import load_dataset

_ALLOWED_EXTENSIONS = {".csv", ".parquet"}


def storage_root(override: str | Path | None = None) -> Path:
    root = Path(override or os.getenv("DATA_STORAGE_DIR", "./data/uploads")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise ValueError("Only CSV and Parquet files are supported.")
    return suffix


def save_and_inspect(content: bytes, original_filename: str, root: str | Path):
    suffix = validate_extension(original_filename)
    if not content:
        raise ValueError("Uploaded file is empty.")
    key = f"{uuid.uuid4().hex}{suffix}"
    path = storage_root(root) / key
    path.write_bytes(content)
    try:
        dataframe = load_dataset(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise ValueError("The uploaded file could not be parsed as a valid dataset.") from exc
    return key, dataframe, path


def resolve_storage_key(key: str, root: str | Path) -> Path:
    root_path = storage_root(root)
    if not key or Path(key).name != key or key in {".", ".."}:
        raise ValueError("Invalid storage key.")
    path = (root_path / key).resolve()
    if path.parent != root_path:
        raise ValueError("Invalid storage key.")
    return path
