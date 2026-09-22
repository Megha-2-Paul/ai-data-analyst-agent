"""Command-line interface for the Stage 1 analytical engine."""

import argparse
import json
from pathlib import Path

import polars as pl

from .agent import AnalystAgent
from .analysis import aggregate, correlation, describe, time_series
from .ingestion import load_dataset
from .llm_planner import OpenAIPlanner
from .planner import plan_query
from .profiling import profile_dataset
from .quality import quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent - Stage 1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("profile", "quality"):
        sub = subparsers.add_parser(command)
        sub.add_argument("path", type=Path, help="Path to a CSV or Parquet dataset")

    plan = subparsers.add_parser("plan")
    plan.add_argument("path", type=Path)
    plan.add_argument("question")

    ask = subparsers.add_parser("ask")
    ask.add_argument("path", type=Path)
    ask.add_argument("question", help="Natural-language analytical question")
    ask.add_argument(
        "--json",
        action="store_true",
        help="Print the complete structured agent response as JSON.",
    )
    ask.add_argument(
        "--planner",
        choices=["deterministic", "openai"],
        default="deterministic",
        help="Planner backend. OpenAI is opt-in and requires the optional llm dependency and API key.",
    )

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
    df = load_dataset(args.path)

    if args.command == "ask":
        planner = OpenAIPlanner() if args.planner == "openai" else plan_query
        response = AnalystAgent(planner=planner).ask(args.question, df)
        result = response.to_dict()
        if not args.json:
            print(response.answer.render_text() if response.answer else "")
            if response.answer and response.answer.evidence:
                print("\nEvidence:")
                for item in response.answer.evidence:
                    context = item.values.get("context")
                    metric = item.values.get("metric")
                    value = item.values.get("value")
                    if context is not None and metric is not None and value is not None:
                        print(f"- {context}: {metric} = {value:.2f}" if isinstance(value, float) else f"- {context}: {metric} = {value}")
                    else:
                        print(f"- {item.claim}")
            if response.answer and response.answer.limitations:
                print("\nLimitations:")
                for limitation in response.answer.limitations:
                    print(f"- {limitation}")
            return
        print(json.dumps(result, indent=2, default=str))
        return
    elif args.command == "plan":
        numeric_columns = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
        datetime_columns = [name for name, dtype in df.schema.items() if dtype in (pl.Date, pl.Datetime)]
        result = plan_query(
            args.question,
            columns=df.columns,
            numeric_columns=numeric_columns,
            datetime_columns=datetime_columns,
        ).to_dict()
    elif args.command == "profile":
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
