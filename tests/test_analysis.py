import polars as pl
import pytest

from data_analyst.analysis import describe_numeric, grouped_summary, value_counts


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "region": ["East", "East", "West", "West"],
            "sales": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_describe_numeric(df):
    result = describe_numeric(df, "sales")
    assert result["count"] == 4
    assert result["mean"] == 25.0
    assert result["median"] == 25.0


def test_grouped_summary(df):
    result = grouped_summary(df, "region", "sales", "sum")
    assert result.to_dict(as_series=False) == {
        "region": ["East", "West"],
        "sales_sum": [30.0, 70.0],
    }


def test_value_counts(df):
    result = value_counts(df, "region")
    assert result.columns == ["region", "count"]
    assert result["count"].sum() == 4


def test_missing_column(df):
    with pytest.raises(KeyError):
        describe_numeric(df, "missing")
