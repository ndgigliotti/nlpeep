from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag-viewer",
        description="A beautiful TUI for viewing RAG pipeline data",
    )
    parser.add_argument("file", type=Path, help="Path to a JSONL file")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a .rag-viewer.toml config file",
    )
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    from rag_viewer.app import RagViewerApp

    app = RagViewerApp(path=args.file, config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
