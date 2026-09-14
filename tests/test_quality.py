from datetime import datetime

import polars as pl
import pytest

from data_analyst.quality import quality_report


def test_quality_report_detects_nulls_duplicates_and_outliers():
    df = pl.DataFrame({"value": [1, 1, 2, 100, None]})
    report = quality_report(df)

    assert report["row_count"] == 5
    assert report["null_counts"] == {"value": 1}
    assert report["duplicate_row_count"] == 1
    assert report["outlier_counts_iqr"]["value"] >= 1
    assert report["warnings"]


def test_quality_report_detects_configured_non_negative_violation():
    df = pl.DataFrame({"fare": [10.0, -2.0, 5.0, None]})
    report = quality_report(df, non_negative_columns=["fare"])

    finding = next(item for item in report["findings"] if item["check"] == "NON_NEGATIVE")
    assert finding["severity"] == "ERROR"
    assert finding["affected_rows"] == 1
    assert finding["affected_pct"] == 25.0
    assert finding["examples"] == [-2.0]


def test_quality_report_detects_datetime_order_violation():
    df = pl.DataFrame(
        {
            "pickup": [datetime(2025, 1, 1, 10), datetime(2025, 1, 1, 12)],
            "dropoff": [datetime(2025, 1, 1, 11), datetime(2025, 1, 1, 11)],
        }
    )
    report = quality_report(df, datetime_order_pairs=[("pickup", "dropoff")])

    finding = next(item for item in report["findings"] if item["check"] == "DATETIME_ORDER")
    assert finding["severity"] == "ERROR"
    assert finding["affected_rows"] == 1


def test_quality_report_detects_unexpected_categories():
    df = pl.DataFrame({"payment_type": [1, 2, 9, None]})
    report = quality_report(df, expected_categories={"payment_type": [1, 2, 3, 4, 5, 6]})

    finding = next(item for item in report["findings"] if item["check"] == "EXPECTED_CATEGORY")
    assert finding["severity"] == "WARNING"
    assert finding["affected_rows"] == 1
    assert finding["examples"] == [9]


def test_quality_report_detects_maximum_value_violation():
    df = pl.DataFrame({"distance": [1.0, 5.0, 12.0, None]})
    report = quality_report(df, maximum_values={"distance": 10.0})

    finding = next(item for item in report["findings"] if item["check"] == "MAXIMUM_VALUE")
    assert finding["severity"] == "WARNING"
    assert finding["affected_rows"] == 1


def test_quality_report_detects_date_range_violation():
    df = pl.DataFrame(
        {
            "pickup": [
                datetime(2025, 1, 1),
                datetime(2025, 2, 1),
                datetime(2025, 1, 15),
            ]
        }
    )
    report = quality_report(
        df,
        date_ranges={
            "pickup": (datetime(2025, 1, 1), datetime(2025, 1, 31, 23, 59, 59))
        },
    )

    finding = next(item for item in report["findings"] if item["check"] == "DATE_RANGE")
    assert finding["severity"] == "WARNING"
    assert finding["affected_rows"] == 1


def test_quality_report_rejects_missing_validation_column():
    df = pl.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match="missing column"):
        quality_report(df, non_negative_columns=["does_not_exist"])
