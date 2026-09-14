"""Command-line interface for the Stage 1 analytical engine."""

import argparse
import json
from pathlib import Path

from .ingestion import load_dataset
from .profiling import profile_dataset
from .quality import quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent - Stage 1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("profile", "quality"):
        sub = subparsers.add_parser(command)
        sub.add_argument("path", type=Path, help="Path to a CSV or Parquet dataset")

    args = parser.parse_args()
    df = load_dataset(args.path)
    result = profile_dataset(df) if args.command == "profile" else quality_report(df)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
