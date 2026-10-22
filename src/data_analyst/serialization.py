"""JSON-safe conversion helpers for analytical result payloads."""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import polars as pl


def json_safe(value: Any) -> Any:
    """Convert analytical values into strict JSON-compatible Python values.

    Dates become ISO-8601 strings, non-finite floats become null, Polars
    containers become ordinary lists/dictionaries, and scalar objects exposing
    ``item()`` (for example NumPy scalars) are converted to their Python value.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, pl.DataFrame):
        return json_safe(value.to_dicts())
    if isinstance(value, pl.Series):
        return json_safe(value.to_list())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return json_safe(converted)

    # Preserve the ability to serialize third-party scalar types without
    # allowing an unexpected object to break the entire response payload.
    return str(value)
