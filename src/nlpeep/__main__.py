from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="peep",
        description="A beautiful TUI for inspecting NLP and RAG pipeline data",
    )
    parser.add_argument(
        "file", type=Path, help="Path to a data file (JSONL, JSON, CSV, TSV, Parquet)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a .nlpeep.toml config file",
    )
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    from nlpeep.app import NLPeepApp

    app = NLPeepApp(path=args.file, config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
