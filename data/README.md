# Local datasets

External datasets are intentionally excluded from version control.

## NYC TLC Stage 1

1. Open the official TLC Trip Record Data page:
   https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. Download one monthly **Yellow Taxi Trip Record** Parquet file.
3. Place it in this directory.
4. Run the Stage 1 CLI against the file.

Example:

```bash
python -m data_analyst.cli profile data/yellow_tripdata_YYYY-MM.parquet
python -m data_analyst.cli quality data/yellow_tripdata_YYYY-MM.parquet
```

The exact month is deliberately chosen at development time so the repository does not embed a large external file.
