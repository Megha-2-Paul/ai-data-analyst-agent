"""Structured result objects for explainable analysis."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    """A traceable analytical result ready for later agent consumption."""

    operation: str
    input_columns: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    calculation: str = ""
    result: Any = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
