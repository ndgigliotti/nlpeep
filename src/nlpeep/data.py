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

    def maybe_assemble_traces(self) -> RecordStore:
        """Detect multi-span trace data and assemble spans into composite records.

        Scans top-level fields for candidate group-ID fields (string/int values
        with cardinality less than the record count). If groups have structurally
        heterogeneous records (different top-level key sets), assembles each
        group into a single composite record by merging child span data into
        the root span.

        Returns a new RecordStore with assembled records, or self unchanged if
        trace detection does not fire.
        """
        if len(self.records) < 2:
            return self

        # Step 1: Find candidate group-ID fields.
        # A candidate has string/int values in all records, and cardinality
        # strictly less than the record count.
        top_keys: set[str] = set()
        for record in self.records:
            top_keys.update(record.data.keys())

        candidates: list[str] = []
        for key in top_keys:
            values: list[str | int] = []
            valid = True
            for record in self.records:
                val = record.data.get(key)
                if not isinstance(val, (str, int)) or isinstance(val, bool):
                    valid = False
                    break
                values.append(val)
            if not valid or len(values) != len(self.records):
                continue
            unique = set(values)
            # Cardinality must be less than record count (some records share values)
            # and at least one group must have more than 1 record
            if len(unique) < len(self.records) and len(unique) < len(values):
                candidates.append(key)

        if not candidates:
            return self

        # Step 2: For each candidate, check structural heterogeneity within groups.
        best_field: str | None = None
        best_groups: dict[str | int, list[Record]] = {}

        for field_name in candidates:
            groups: dict[str | int, list[Record]] = {}
            for record in self.records:
                gid = record.data[field_name]
                groups.setdefault(gid, []).append(record)

            # Only consider groups with more than 1 record
            multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
            if not multi_groups:
                continue

            # Check structural heterogeneity: do records within groups have
            # different sets of top-level keys?
            heterogeneous = False
            for group_records in multi_groups.values():
                key_sets = [frozenset(r.data.keys()) for r in group_records]
                if len(set(key_sets)) > 1:
                    heterogeneous = True
                    break

            if heterogeneous:
                best_field = field_name
                best_groups = groups
                break

        if best_field is None:
            return self

        # Step 3: Assemble each group into a composite record.
        composites: list[Record] = []
        for idx, (gid, group_records) in enumerate(best_groups.items()):
            if len(group_records) == 1:
                # Single-record groups stay as-is
                composites.append(Record(index=idx, data=group_records[0].data.copy()))
                continue

            composite_data = _assemble_trace_group(group_records)
            composites.append(Record(index=idx, data=composite_data))

        return RecordStore(records=composites, skipped=self.skipped, path=self.path)


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


# -- Trace assembly helpers ---------------------------------------------------

_PARENT_FIELDS = ("parent_run_id", "parent_id", "parent_span_id")


def _find_root_span(records: list[Record]) -> Record:
    """Find the root span in a list of trace records.

    The root is the record whose parent reference field is null/None.
    If no clear root is found, returns the first record by position.
    """
    for record in records:
        for pf in _PARENT_FIELDS:
            val = record.data.get(pf)
            # Explicit null or absent field signals root
            if pf in record.data and val is None:
                return record
    return records[0]


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively copy a dict, handling nested dicts and lists."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = _deep_copy_list(v)
        else:
            result[k] = v
    return result


def _deep_copy_list(lst: list[Any]) -> list[Any]:
    """Recursively copy a list."""
    result: list[Any] = []
    for item in lst:
        if isinstance(item, dict):
            result.append(_deep_copy_dict(item))
        elif isinstance(item, list):
            result.append(_deep_copy_list(item))
        else:
            result.append(item)
    return result


def _is_substantial_value(val: Any) -> bool:
    """Check if a value is substantial content worth promoting from a child span.

    Substantial means: a list of dicts (documents, messages, generations),
    or a non-trivial nested dict.
    """
    if isinstance(val, list) and val:
        if any(isinstance(item, dict) for item in val):
            return True
    return False


def _promote_child_values(
    base: dict[str, Any], child: dict[str, Any], prefix: str = "",
) -> None:
    """Walk child data and promote substantial values not present in base.

    Copies list-of-dict values (documents, messages, etc.) from child into
    base under the same nested path, if the base lacks that value.
    """
    for key, val in child.items():
        if isinstance(val, dict):
            if key not in base:
                base[key] = {}
            if isinstance(base.get(key), dict):
                _promote_child_values(base[key], val, prefix=f"{prefix}.{key}" if prefix else key)
        elif _is_substantial_value(val):
            # Only promote if the base does not already have this value
            if key not in base or base[key] is None:
                base[key] = _deep_copy_list(val) if isinstance(val, list) else val
            elif isinstance(base[key], list) and not base[key]:
                base[key] = _deep_copy_list(val) if isinstance(val, list) else val


def _assemble_trace_group(records: list[Record]) -> dict[str, Any]:
    """Assemble a list of span records into a single composite record.

    1. Find the root span and use its data as the base.
    2. Promote substantial values from child spans.
    3. Add _trace_spans with all original span data.
    """
    root = _find_root_span(records)
    composite = _deep_copy_dict(root.data)

    # Promote child span values into the composite
    for record in records:
        if record is root:
            continue
        _promote_child_values(composite, record.data)

    # Build _trace_spans: all spans ordered by start_time if available,
    # otherwise by original position
    def _sort_key(r: Record) -> tuple[str, int]:
        st = r.data.get("start_time", "")
        return (st if isinstance(st, str) else "", r.index)

    sorted_records = sorted(records, key=_sort_key)
    spans: list[dict[str, Any]] = []
    for record in sorted_records:
        span = _deep_copy_dict(record.data)
        name = record.data.get("name")
        if name is not None:
            span["_span_name"] = name
        spans.append(span)

    composite["_trace_spans"] = spans
    return composite
