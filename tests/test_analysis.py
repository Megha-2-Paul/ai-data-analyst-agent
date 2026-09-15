from datetime import datetime

import polars as pl
import pytest

from data_analyst.analysis import aggregate, correlation, describe, describe_numeric, grouped_summary, time_series, value_counts


@pytest.fixture
def df():
    return pl.DataFrame({"region": ["East", "East", "West", "West"], "sales": [10.0, 20.0, 30.0, 40.0]})


def test_describe_numeric(df):
    result = describe_numeric(df, "sales")
    assert result["count"] == 4
    assert result["mean"] == 25.0
    assert result["median"] == 25.0


def test_grouped_summary(df):
    result = grouped_summary(df, "region", "sales", "sum")
    assert result.to_dict(as_series=False) == {"region": ["East", "West"], "sales_sum": [30.0, 70.0]}


def test_value_counts(df):
    result = value_counts(df, "region")
    assert result.columns == ["region", "count"]
    assert result["count"].sum() == 4


def test_missing_column(df):
    with pytest.raises(KeyError):
        describe_numeric(df, "missing")


def test_describe_returns_machine_readable_statistics(df):
    result = describe(df)
    sales = next(item for item in result["result"] if item["column"] == "sales")
    assert result["analysis"] == "descriptive_statistics"
    assert sales["count"] == 4
    assert sales["p25"] == 17.5
    assert sales["p75"] == 32.5
    assert sales["unique"] == 4


def test_aggregate_supports_multiple_operations(df):
    for operation, expected in [("count", 2), ("sum", 30.0), ("mean", 15.0), ("median", 15.0), ("min", 10.0), ("max", 20.0), ("std", 7.0710678118654755)]:
        result = aggregate(df, group_by=["region"], metric="sales", agg=operation)
        first = result["result"][0]["sales"]
        assert first == pytest.approx(expected)


def test_aggregate_rejects_unknown_operation(df):
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        aggregate(df, group_by=["region"], metric="sales", agg="mode")


def test_correlation_returns_numeric_matrix(df):
    result = correlation(df, ["sales"])
    assert result["analysis"] == "correlation"
    assert result["result"][0]["sales"] == pytest.approx(1.0)


def test_time_series_groups_rows_by_hour():
    df = pl.DataFrame({"timestamp": [datetime(2025, 1, 1, 10, 5), datetime(2025, 1, 1, 10, 45), datetime(2025, 1, 1, 11, 10)], "sales": [10.0, 20.0, 30.0]})
    result = time_series(df, datetime_column="timestamp", frequency="1h")
    assert result["analysis"] == "time_series"
    assert [row["row_count"] for row in result["result"]] == [2, 1]


def test_time_series_can_aggregate_metric():
    df = pl.DataFrame({"timestamp": [datetime(2025, 1, 1, 10), datetime(2025, 1, 1, 10, 30)], "sales": [10.0, 30.0]})
    result = time_series(df, datetime_column="timestamp", frequency="1h", metric="sales", agg="mean")
    assert result["result"][0]["sales"] == 20.0
