"""Command-line interface for the Stage 1 analytical engine."""

import argparse
import json
from pathlib import Path

import polars as pl

from .analysis import aggregate, correlation, describe, time_series
from .ingestion import load_dataset, scan_dataset
from .planner import plan_query
from .profiling import profile_dataset
from .quality import quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent - Stage 1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("profile", "quality"):
        sub = subparsers.add_parser(command)
        sub.add_argument("path", type=Path, help="Path to a CSV or Parquet dataset")

    sub = subparsers.add_parser("plan")
    sub.add_argument("path", type=Path, help="Path to a CSV or Parquet dataset")
    sub.add_argument("question", help="Natural-language analytical question")

    sub = subparsers.add_parser("describe")
    sub.add_argument("path", type=Path)
    sub.add_argument("--columns", nargs="+")

    sub = subparsers.add_parser("groupby")
    sub.add_argument("path", type=Path)
    sub.add_argument("--group-by", nargs="+", required=True)
    sub.add_argument("--metric", required=True)
    sub.add_argument("--agg", choices=["count", "sum", "mean", "median", "min", "max", "std"], default="mean")

    sub = subparsers.add_parser("correlation")
    sub.add_argument("path", type=Path)
    sub.add_argument("--columns", nargs="+")

    sub = subparsers.add_parser("time-analysis")
    sub.add_argument("path", type=Path)
    sub.add_argument("--datetime-column", required=True)
    sub.add_argument("--frequency", default="1d", help="Polars duration, e.g. 1d, 1h, 1w")
    sub.add_argument("--metric")
    sub.add_argument("--agg", choices=["count", "sum", "mean", "median", "min", "max", "std"], default="count")

    args = parser.parse_args()

    if args.command == "plan":
        schema = scan_dataset(args.path).collect_schema()
        numeric_columns = [name for name, dtype in schema.items() if dtype.is_numeric()]
        datetime_columns = [
            name for name, dtype in schema.items()
            if dtype in (pl.Date, pl.Datetime)
        ]
        result = plan_query(
            args.question,
            columns=list(schema.keys()),
            numeric_columns=numeric_columns,
            datetime_columns=datetime_columns,
        ).to_dict()
        print(json.dumps(result, indent=2, default=str))
        return

    df = load_dataset(args.path)
    if args.command == "profile":
        result = profile_dataset(df)
    elif args.command == "quality":
        result = quality_report(df)
    elif args.command == "describe":
        result = describe(df, args.columns)
    elif args.command == "groupby":
        result = aggregate(df, group_by=args.group_by, metric=args.metric, agg=args.agg)
    elif args.command == "correlation":
        result = correlation(df, args.columns)
    else:
        result = time_series(df, datetime_column=args.datetime_column, frequency=args.frequency, metric=args.metric, agg=args.agg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
