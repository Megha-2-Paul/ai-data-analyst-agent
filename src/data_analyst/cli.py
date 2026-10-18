"""Command-line interface for the analytical engine."""

import argparse
import json
from pathlib import Path

import polars as pl

from .agent import AnalystAgent
from .analysis import aggregate, correlation, describe, time_series
from .ingestion import load_dataset, scan_dataset
from .llm_planner import OpenAIPlanner
from .providers import GeminiPlanner, GroqPlanner
from .planner import plan_query
from .preparation import prepare_dataset
from .profiling import profile_dataset
from .quality import quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("profile", "quality", "prepare"):
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
        "--prepare",
        action="store_true",
        help="Apply safe automatic data preparation before analysis.",
    )
    ask.add_argument(
        "--planner",
        choices=["deterministic", "openai", "groq", "gemini"],
        default="deterministic",
        help="Planner backend. AI providers are opt-in and require their SDK and API key.",
    )

    sub = subparsers.add_parser("describe")
    sub.add_argument("path", type=Path)
    sub.add_argument("--columns", nargs="+")

    sub = subparsers.add_parser("groupby")
    sub.add_argument("path", type=Path)
    sub.add_argument("--group-by", nargs="+", required=True)
    sub.add_argument("--metric", required=True)
    sub.add_argument(
        "--agg",
        choices=["count", "sum", "mean", "median", "min", "max", "std"],
        default="mean",
    )

    sub = subparsers.add_parser("correlation")
    sub.add_argument("path", type=Path)
    sub.add_argument("--columns", nargs="+")

    sub = subparsers.add_parser("time-analysis")
    sub.add_argument("path", type=Path)
    sub.add_argument("--datetime-column", required=True)
    sub.add_argument("--frequency", default="1d", help="Polars duration, e.g. 1d, 1h, 1w")
    sub.add_argument("--metric")
    sub.add_argument(
        "--agg",
        choices=["count", "sum", "mean", "median", "min", "max", "std"],
        default="count",
    )

    ml_predict = subparsers.add_parser("ml-predict", help="Train and evaluate an optional baseline ML model.")
    ml_predict.add_argument("path", type=Path)
    ml_predict.add_argument("--target", required=True)
    ml_predict.add_argument("--features", nargs="+", required=True)
    ml_predict.add_argument("--task", choices=["regression", "classification"], default="regression")
    ml_predict.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()

    # Planning needs only the schema. Keep this path lazy so large datasets
    # are not fully loaded merely to construct a QueryPlan.
    if args.command == "plan":
        schema = scan_dataset(args.path).collect_schema()
        numeric_columns = [
            name for name, dtype in schema.items() if dtype.is_numeric()
        ]
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

    preparation = None
    if args.command == "prepare" or getattr(args, "prepare", False):
        preparation = prepare_dataset(df)
        df = preparation.dataframe

    if args.command == "ask":
        planner = {"deterministic": plan_query, "openai": OpenAIPlanner, "groq": GroqPlanner, "gemini": GeminiPlanner}[args.planner]
        if args.planner != "deterministic":
            planner = planner()
        response = AnalystAgent(planner=planner).ask(args.question, df)
        result = response.to_dict()
        if preparation is not None:
            result["preparation"] = preparation.to_dict()

        if not args.json:
            if preparation is not None:
                print(
                    f"Preparation: applied "
                    f"{len(preparation.cleaning.applied_steps)} safe step(s)."
                )
            print(response.answer.render_text() if response.answer else "")
            if response.answer and response.answer.evidence:
                print("\nEvidence:")
                for item in response.answer.evidence:
                    context = item.values.get("context")
                    metric = item.values.get("metric")
                    value = item.values.get("value")
                    if context is not None and metric is not None and value is not None:
                        if isinstance(value, float):
                            print(f"- {context}: {metric} = {value:.2f}")
                        else:
                            print(f"- {context}: {metric} = {value}")
                    else:
                        print(f"- {item.claim}")
            if response.answer and response.answer.limitations:
                print("\nLimitations:")
                for limitation in response.answer.limitations:
                    print(f"- {limitation}")
            return

        print(json.dumps(result, indent=2, default=str))
        return

    if args.command == "ml-predict":
        from .answer import generate_answer
        from .ml import train_predictive_model
        result = train_predictive_model(
            df,
            target=args.target,
            features=args.features,
            task=args.task,
            test_size=args.test_size,
        )
        result["answer"] = generate_answer(result).to_dict()
        print(json.dumps(result, indent=2, default=str, allow_nan=False))
        return

    if args.command == "profile":
        result = profile_dataset(df)
    elif args.command == "quality":
        result = quality_report(df)
    elif args.command == "prepare":
        result = preparation.to_dict()
    elif args.command == "describe":
        result = describe(df, args.columns)
    elif args.command == "groupby":
        result = aggregate(
            df,
            group_by=args.group_by,
            metric=args.metric,
            agg=args.agg,
        )
    elif args.command == "correlation":
        result = correlation(df, args.columns)
    else:
        result = time_series(
            df,
            datetime_column=args.datetime_column,
            frequency=args.frequency,
            metric=args.metric,
            agg=args.agg,
        )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
