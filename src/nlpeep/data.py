from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class Record:
    index: int
    data: dict[str, Any]

    def get_path(self, path: str) -> Any:
        """Resolve a dot-notation path against this record's data.

        Uses greedy matching so that dotted key names (e.g. Phoenix
        OpenInference ``"input.value"``) are resolved correctly.
        """
        parts = path.split(".")
        current: Any = self.data
        i = 0
        while i < len(parts):
            if isinstance(current, dict):
                matched = False
                for j in range(len(parts), i, -1):
                    candidate = ".".join(parts[i:j])
                    if candidate in current:
                        current = current[candidate]
                        i = j
                        matched = True
                        break
                if not matched:
                    return None
            elif isinstance(current, list):
                try:
                    current = current[int(parts[i])]
                    i += 1
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current

    def label(self, label_path: str | None = None) -> str:
        """Generate a display label for the navigator."""
        if label_path:
            val = self.get_path(label_path)
            if val and isinstance(val, str):
                return _truncate(val, 60)

        # Fallback: find the first short-ish string field
        for key in self.data:
            val = self.data[key]
            if isinstance(val, str) and len(val) < 200:
                return _truncate(val, 60)

        return f"Record {self.index}"


@dataclass
class RecordStore:
    records: list[Record] = field(default_factory=list)
    skipped: int = 0
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> RecordStore:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return cls._load_parquet(path)
        if suffix in (".csv", ".tsv"):
            return cls._load_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
        if suffix == ".json":
            return cls._load_json(path)
        # Default: try JSONL, fall back to JSON array if first line fails
        return cls._load_jsonl(path)

    @classmethod
    def _load_jsonl(cls, path: Path) -> RecordStore:
        records: list[Record] = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        records.append(Record(index=i, data=data))
                    elif isinstance(data, list) and i == 0:
                        # First line parsed as a JSON array -- switch to JSON loader
                        return cls._load_json(path)
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1
        return cls(records=records, skipped=skipped, path=path)

    @classmethod
    def _load_json(cls, path: Path) -> RecordStore:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = [
                Record(index=i, data=item)
                for i, item in enumerate(data)
                if isinstance(item, dict)
            ]
            skipped = len(data) - len(records)
            return cls(records=records, skipped=skipped, path=path)
        if isinstance(data, dict):
            return cls(records=[Record(index=0, data=data)], skipped=0, path=path)
        return cls(records=[], skipped=1, path=path)

    @classmethod
    def _load_csv(cls, path: Path, delimiter: str = ",") -> RecordStore:
        rows = pl.read_csv(path, separator=delimiter).to_dicts()
        records = [Record(index=i, data=row) for i, row in enumerate(rows)]
        return cls(records=records, skipped=0, path=path)

    @classmethod
    def _load_parquet(cls, path: Path) -> RecordStore:
        rows = pl.read_parquet(path).to_dicts()
        records = [Record(index=i, data=row) for i, row in enumerate(rows)]
        return cls(records=records, skipped=0, path=path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Record:
        return self.records[index]

    def search(self, query: str) -> list[int]:
        """Return indices of records containing query as substring in any string value."""
        query_lower = query.lower()
        results = []
        for i, record in enumerate(self.records):
            if _record_matches(record.data, query_lower):
                results.append(i)
        return results

    def field_summary(self) -> dict[str, set[str]]:
        """Return union of all field paths (including nested) with their observed Python type names.

        Nested dict fields are expanded into dot-paths (e.g. ``inputs.question``).
        Lists are not recursed into -- they are treated as leaf values.
        The depth is capped at 4 levels to avoid runaway expansion.
        """
        summary: dict[str, set[str]] = {}
        for record in self.records:
            for dot_path, val in _flatten_for_summary(record.data).items():
                if dot_path not in summary:
                    summary[dot_path] = set()
                summary[dot_path].add(type(val).__name__)
        return summary

    def sample(self, n: int = 20) -> list[Record]:
        """Return up to n records for schema auto-detection."""
        return self.records[:n]


def _truncate(s: str, length: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= length:
        return s
    return s[: length - 1] + "\u2026"


_MAX_SUMMARY_DEPTH = 4


def _flatten_for_summary(data: dict[str, Any], prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Flatten nested dicts into dot-paths, keeping lists as leaf values."""
    result: dict[str, Any] = {}
    for key, val in data.items():
        dot_path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and depth < _MAX_SUMMARY_DEPTH:
            result.update(_flatten_for_summary(val, dot_path, depth + 1))
        else:
            result[dot_path] = val
    return result


def _record_matches(data: Any, query: str) -> bool:
    if isinstance(data, str):
        return query in data.lower()
    if isinstance(data, dict):
        return any(_record_matches(v, query) for v in data.values())
    if isinstance(data, list):
        return any(_record_matches(item, query) for item in data)
    return False
