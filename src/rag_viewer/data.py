from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Record:
    index: int
    data: dict[str, Any]

    def get_path(self, path: str) -> Any:
        """Resolve a dot-notation path against this record's data."""
        current: Any = self.data
        for part in path.split("."):
            if isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
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
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1
        return cls(records=records, skipped=skipped, path=path)

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
        """Return union of all top-level keys with their observed Python type names."""
        summary: dict[str, set[str]] = {}
        for record in self.records:
            for key, val in record.data.items():
                if key not in summary:
                    summary[key] = set()
                summary[key].add(type(val).__name__)
        return summary

    def sample(self, n: int = 20) -> list[Record]:
        """Return up to n records for schema auto-detection."""
        return self.records[:n]


def _truncate(s: str, length: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= length:
        return s
    return s[: length - 1] + "\u2026"


def _record_matches(data: Any, query: str) -> bool:
    if isinstance(data, str):
        return query in data.lower()
    if isinstance(data, dict):
        return any(_record_matches(v, query) for v in data.values())
    if isinstance(data, list):
        return any(_record_matches(item, query) for item in data)
    return False
