"""Stage 4 dataset preparation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from .cleaning import CleaningPlan, CleaningResult, apply_cleaning_plan, build_cleaning_plan
from .profiling import profile_dataset
from .quality import quality_report


@dataclass(frozen=True)
class PreparationResult:
    """Complete preparation trace, preserving both before and after state."""

    original_profile: dict[str, Any]
    original_quality: dict[str, Any]
    plan: CleaningPlan
    cleaning: CleaningResult
    prepared_profile: dict[str, Any]
    prepared_quality: dict[str, Any]

    @property
    def dataframe(self) -> pl.DataFrame:
        return self.cleaning.dataframe

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_profile": self.original_profile,
            "original_quality": self.original_quality,
            "plan": self.plan.to_dict(),
            "cleaning": self.cleaning.to_dict(),
            "prepared_profile": self.prepared_profile,
            "prepared_quality": self.prepared_quality,
        }


def prepare_dataset(
    df: pl.DataFrame,
    *,
    plan: CleaningPlan | None = None,
) -> PreparationResult:
    """Profile, assess, safely clean, and reassess a dataframe.

    The source dataframe is never mutated. Statistical imputation, outlier
    removal, and domain-specific corrections remain recommendations rather
    than automatic transformations.
    """

    original_profile = profile_dataset(df)
    original_quality = quality_report(df)
    selected_plan = plan or build_cleaning_plan(df)
    cleaning = apply_cleaning_plan(df, selected_plan)
    prepared = cleaning.dataframe
    prepared_profile = profile_dataset(prepared)
    prepared_quality = quality_report(prepared)

    return PreparationResult(
        original_profile=original_profile,
        original_quality=original_quality,
        plan=selected_plan,
        cleaning=cleaning,
        prepared_profile=prepared_profile,
        prepared_quality=prepared_quality,
    )
