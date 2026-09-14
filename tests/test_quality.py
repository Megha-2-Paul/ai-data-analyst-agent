import polars as pl

from data_analyst.quality import quality_report


def test_quality_report_detects_nulls_duplicates_and_outliers():
    df = pl.DataFrame({"value": [1, 1, 2, 100, None]})
    report = quality_report(df)

    assert report["row_count"] == 5
    assert report["null_counts"] == {"value": 1}
    assert report["duplicate_row_count"] == 1
    assert report["outlier_counts_iqr"]["value"] >= 1
    assert report["warnings"]
