import polars as pl

from data_analyst.preparation import prepare_dataset


def test_prepare_dataset_returns_before_after_reports_without_mutating_input():
    df = pl.DataFrame(
        {
            "Customer Name": [" Alice ", "", "Bob"],
            "order_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "value": [1.0, None, 100.0],
        }
    )

    result = prepare_dataset(df)

    assert result.dataframe is not df
    assert df.columns == ["Customer Name", "order_date", "value"]
    assert result.original_profile["column_count"] == 3
    assert result.prepared_profile["column_count"] == 3
    assert result.cleaning.applied_steps
    assert result.to_dict()["plan"]["steps"]
    assert "value" in result.prepared_quality["null_counts"]


def test_prepare_dataset_does_not_impute_or_remove_outliers():
    df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 1000.0, None]})

    result = prepare_dataset(df)

    assert result.dataframe.height == 4
    assert result.dataframe["value"].null_count() == 1
    assert result.dataframe["value"].to_list()[-2] == 1000.0
    assert any("outlier" in item.lower() for item in result.cleaning.recommendations)
