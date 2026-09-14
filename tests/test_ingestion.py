import polars as pl
import pytest

from data_analyst.ingestion import load_dataset


def test_load_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("name,value\na,1\nb,2\n", encoding="utf-8")
    df = load_dataset(path)
    assert df.shape == (2, 2)
    assert df.columns == ["name", "value"]


def test_load_parquet(tmp_path):
    path = tmp_path / "sample.parquet"
    pl.DataFrame({"value": [1, 2, 3]}).write_parquet(path)
    df = load_dataset(path)
    assert df.shape == (3, 1)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.parquet")


def test_rejects_unsupported_format(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_dataset(path)
