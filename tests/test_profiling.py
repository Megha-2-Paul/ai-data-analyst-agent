import polars as pl

from data_analyst.profiling import profile_dataset


def test_profile_dataset_reports_shape_nulls_and_duplicates():
    df = pl.DataFrame({"category": ["A", "A", "B"], "value": [1.0, None, 3.0]})
    profile = profile_dataset(df)

    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    assert profile["duplicate_row_count"] == 0

    value_column = next(item for item in profile["columns"] if item["name"] == "value")
    assert value_column["null_count"] == 1
    assert value_column["null_pct"] == 33.3333
