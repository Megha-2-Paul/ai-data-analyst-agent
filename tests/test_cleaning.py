from datetime import datetime

import polars as pl
import pytest

from data_analyst.cleaning import (
    CleaningError,
    CleaningPlan,
    CleaningStep,
    apply_cleaning_plan,
    build_cleaning_plan,
)


def test_build_plan_detects_safe_string_and_datetime_cleanup():
    df = pl.DataFrame(
        {
            "Customer Name": [" Alice ", "", "Bob "],
            "order_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "empty": [None, None, None],
        }
    )

    plan = build_cleaning_plan(df)
    operations = [step.operation for step in plan.steps]

    assert "normalize_column_names" in operations
    assert "trim_strings" in operations
    assert "empty_strings_to_null" in operations
    assert "parse_datetime" in operations
    assert "drop_empty_columns" in operations
    assert any("Missing values" in item for item in plan.recommendations)


def test_apply_cleaning_plan_preserves_source_and_cleans_values():
    df = pl.DataFrame(
        {
            "Customer Name": [" Alice ", " ", "Bob"],
            "order_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        }
    )

    result = apply_cleaning_plan(df, build_cleaning_plan(df))
    cleaned = result.dataframe

    assert df.columns == ["Customer Name", "order_date"]
    assert cleaned.columns == ["customer_name", "order_date"]
    assert cleaned["customer_name"].to_list() == ["Alice", None, "Bob"]
    assert cleaned["order_date"].dtype == pl.Datetime
    assert cleaned["order_date"].to_list()[0] == datetime(2025, 1, 1)


def test_plan_rejects_unsafe_operation():
    df = pl.DataFrame({"value": [1, 2, 3]})
    plan = CleaningPlan(
        steps=[CleaningStep("step_1", "mean_imputation", columns=["value"])]
    )

    with pytest.raises(CleaningError, match="Unsupported automatic cleaning"):
        apply_cleaning_plan(df, plan)


def test_duplicate_rows_are_recommendation_not_automatic_cleaning():
    df = pl.DataFrame({"value": [1, 1, 2]})
    plan = build_cleaning_plan(df)

    assert not any(step.operation == "drop_duplicates" for step in plan.steps)
    assert any("duplicate" in item.lower() for item in plan.recommendations)
