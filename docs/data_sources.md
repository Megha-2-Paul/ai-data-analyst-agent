# Real-world data sources

The project is designed around **real data published by original organizations**. Synthetic data is not used as the primary demonstration dataset.

## Stage 1: NYC TLC Yellow Taxi Trip Records

**Publisher:** New York City Taxi and Limousine Commission (TLC)

**Official source:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Use one monthly Yellow Taxi Trip Record Parquet file for local Stage 1 development. Do not commit the downloaded file to GitHub.

The records include operational fields such as pickup/drop-off timestamps and locations, passenger count, trip distance, fares, payment information and related trip attributes.

## Later validation: UDISE+

**Publisher:** Government of India Open Government Data platform / Ministry of Education

**Official portal:** https://www.data.gov.in/

A suitable UDISE+ resource will be selected for cross-domain validation after the Stage 1 engine is stable.

## Later integration: World Bank World Development Indicators

**Publisher:** World Bank

**Official portal:** https://data.worldbank.org/

The World Bank API will be integrated later so the system can retrieve real longitudinal indicators programmatically.

## Data handling principles

- Preserve attribution to the original publisher.
- Do not commit large external datasets.
- Record the source URL and dataset description.
- Keep transformations reproducible in code.
- Keep dataset-specific logic out of the core analytical engine.
