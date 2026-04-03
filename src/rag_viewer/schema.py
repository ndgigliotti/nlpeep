from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FieldRole(str, Enum):
    QUERY = "query"
    RESPONSE = "response"
    DOCUMENTS = "documents"
    METRICS = "metrics"
    TRACE = "trace"
    GROUND_TRUTH = "ground_truth"
    ID = "id"
    METADATA = "metadata"
    UNMAPPED = "unmapped"

    @property
    def display_name(self) -> str:
        return {
            FieldRole.QUERY: "Query",
            FieldRole.RESPONSE: "Response",
            FieldRole.DOCUMENTS: "Documents",
            FieldRole.METRICS: "Metrics",
            FieldRole.TRACE: "Trace",
            FieldRole.GROUND_TRUTH: "Ground Truth",
            FieldRole.ID: "ID",
            FieldRole.METADATA: "Metadata",
            FieldRole.UNMAPPED: "Details",
        }[self]


# Patterns for name-based auto-detection, ordered by specificity
_ROLE_PATTERNS: list[tuple[FieldRole, re.Pattern[str]]] = [
    (FieldRole.ID, re.compile(r"^(id|_id|record_id|example_id|idx|uuid)$", re.I)),
    (FieldRole.QUERY, re.compile(r"^(query|question|input|prompt|user_input|q|user_query|search_query)$", re.I)),
    (FieldRole.RESPONSE, re.compile(r"^(response|answer|output|result|generation|completion|predicted|llm_response|assistant|reply)$", re.I)),
    (FieldRole.GROUND_TRUTH, re.compile(r"^(ground_truth|expected|gold|reference|target|label|correct_answer)$", re.I)),
    (FieldRole.DOCUMENTS, re.compile(r"^(documents|docs|retrieved_documents|contexts|passages|chunks|results|retrieved|retrieved_passages|context|sources|retrieved_chunks)$", re.I)),
    (FieldRole.METRICS, re.compile(r"^(metrics|scores|evaluation|eval_results|eval|evaluations)$", re.I)),
    (FieldRole.TRACE, re.compile(r"^(trace|steps|tool_calls|llm_calls|messages|chain|actions|trajectory|history|tool_use)$", re.I)),
    (FieldRole.METADATA, re.compile(r"^(metadata|meta|info|config|params|settings|extras)$", re.I)),
]


@dataclass
class FieldMapping:
    json_path: str
    role: FieldRole
    display_name: str = ""
    sub_fields: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.role.display_name


@dataclass
class SchemaMapping:
    mappings: list[FieldMapping] = field(default_factory=list)

    def get_mapping(self, role: FieldRole) -> FieldMapping | None:
        for m in self.mappings:
            if m.role == role:
                return m
        return None

    def get_all_for_role(self, role: FieldRole) -> list[FieldMapping]:
        return [m for m in self.mappings if m.role == role]

    def resolve(self, record_data: dict[str, Any], role: FieldRole) -> Any:
        mapping = self.get_mapping(role)
        if not mapping:
            return None
        return _resolve_path(record_data, mapping.json_path)

    def active_roles(self) -> list[FieldRole]:
        seen: set[FieldRole] = set()
        roles: list[FieldRole] = []
        for m in self.mappings:
            if m.role not in seen and m.role != FieldRole.UNMAPPED:
                seen.add(m.role)
                roles.append(m.role)
        return roles

    def unmapped_fields(self, record_data: dict[str, Any]) -> dict[str, Any]:
        mapped_paths = {m.json_path for m in self.mappings if m.role != FieldRole.UNMAPPED}
        result = {}
        for key, val in record_data.items():
            if key not in mapped_paths:
                result[key] = val
        return result

    def label_path(self) -> str | None:
        for role in (FieldRole.QUERY, FieldRole.ID):
            m = self.get_mapping(role)
            if m:
                return m.json_path
        return None

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"mapping": {}, "display": {}}
        for m in self.mappings:
            if m.role == FieldRole.UNMAPPED:
                continue
            config["mapping"][m.role.value] = m.json_path
            if m.sub_fields:
                config["mapping"][f"{m.role.value}_fields"] = m.sub_fields
            if m.display_name != m.role.display_name:
                config["display"][m.role.value] = m.display_name
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SchemaMapping:
        mapping_section = config.get("mapping", {})
        display_section = config.get("display", {})
        mappings: list[FieldMapping] = []

        for role in FieldRole:
            if role == FieldRole.UNMAPPED:
                continue
            path = mapping_section.get(role.value)
            if path and isinstance(path, str):
                sub_fields = mapping_section.get(f"{role.value}_fields", {})
                display_name = display_section.get(role.value, role.display_name)
                mappings.append(FieldMapping(
                    json_path=path,
                    role=role,
                    display_name=display_name,
                    sub_fields=sub_fields if isinstance(sub_fields, dict) else {},
                    confidence=1.0,
                ))

        return cls(mappings=mappings)

    @classmethod
    def auto_detect(cls, records: list[Any]) -> SchemaMapping:
        if not records:
            return cls()

        # Collect all top-level keys across sample records
        key_counts: dict[str, int] = {}
        key_samples: dict[str, list[Any]] = {}
        total = len(records)

        for record in records:
            data = record.data if hasattr(record, "data") else record
            if not isinstance(data, dict):
                continue
            for key, val in data.items():
                key_counts[key] = key_counts.get(key, 0) + 1
                if key not in key_samples:
                    key_samples[key] = []
                if len(key_samples[key]) < 5:
                    key_samples[key].append(val)

        mappings: list[FieldMapping] = []
        claimed_roles: set[FieldRole] = set()

        # Pass 1: Name-based matching
        for key in key_counts:
            for role, pattern in _ROLE_PATTERNS:
                if role in claimed_roles:
                    continue
                if pattern.match(key):
                    presence = key_counts[key] / total
                    confidence = 0.9 * presence
                    mapping = FieldMapping(
                        json_path=key,
                        role=role,
                        confidence=confidence,
                    )
                    # For documents role, try to detect sub-fields
                    if role == FieldRole.DOCUMENTS:
                        mapping.sub_fields = _detect_doc_subfields(key_samples.get(key, []))
                    mappings.append(mapping)
                    claimed_roles.add(role)
                    break

        # Pass 2: Structural analysis for unclaimed roles
        for key, samples in key_samples.items():
            if any(m.json_path == key for m in mappings):
                continue

            role = _structural_role(key, samples)
            if role != FieldRole.UNMAPPED and role not in claimed_roles:
                presence = key_counts[key] / total
                mapping = FieldMapping(
                    json_path=key,
                    role=role,
                    confidence=0.5 * presence,
                )
                if role == FieldRole.DOCUMENTS:
                    mapping.sub_fields = _detect_doc_subfields(samples)
                mappings.append(mapping)
                claimed_roles.add(role)

        return cls(mappings=mappings)


def _resolve_path(data: Any, path: str) -> Any:
    current = data
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


def _structural_role(key: str, samples: list[Any]) -> FieldRole:
    """Infer a role from the shape of sampled values."""
    if not samples:
        return FieldRole.UNMAPPED

    # Check if it looks like documents (list of dicts with text+score-like fields)
    if all(isinstance(s, list) for s in samples):
        flat_items = [item for s in samples for item in s if isinstance(item, dict)]
        if flat_items and _looks_like_docs(flat_items):
            return FieldRole.DOCUMENTS
        if flat_items and _looks_like_trace(flat_items):
            return FieldRole.TRACE

    # Check if it looks like metrics (dict of numbers)
    if all(isinstance(s, dict) for s in samples):
        if all(
            all(isinstance(v, (int, float)) for v in s.values())
            for s in samples
            if s
        ):
            return FieldRole.METRICS

    return FieldRole.UNMAPPED


def _looks_like_docs(items: list[dict[str, Any]]) -> bool:
    """Check if list items look like retrieved documents."""
    has_text = False
    has_score = False
    for item in items[:5]:
        keys_lower = {k.lower() for k in item}
        if keys_lower & {"text", "content", "passage", "chunk", "document", "page_content"}:
            has_text = True
        if keys_lower & {"score", "relevance", "similarity", "distance", "rank"}:
            has_score = True
    return has_text and has_score


def _looks_like_trace(items: list[dict[str, Any]]) -> bool:
    """Check if list items look like trace steps."""
    for item in items[:5]:
        keys_lower = {k.lower() for k in item}
        if keys_lower & {"role", "action", "step", "tool", "type", "function"}:
            return True
    return False


def _detect_doc_subfields(samples: list[Any]) -> dict[str, str]:
    """Try to detect which sub-fields of document objects are text, score, source."""
    sub_fields: dict[str, str] = {}
    items: list[dict[str, Any]] = []
    for s in samples:
        if isinstance(s, list):
            items.extend(item for item in s if isinstance(item, dict))

    if not items:
        return sub_fields

    sample = items[0]
    text_keys = {"text", "content", "passage", "chunk", "document", "page_content", "body"}
    score_keys = {"score", "relevance", "similarity", "distance", "rank", "relevance_score"}
    source_keys = {"source", "url", "title", "doc_id", "filename", "path", "name", "id"}

    for key in sample:
        kl = key.lower()
        if kl in text_keys and "text" not in sub_fields:
            sub_fields["text"] = key
        elif kl in score_keys and "score" not in sub_fields:
            sub_fields["score"] = key
        elif kl in source_keys and "source" not in sub_fields:
            sub_fields["source"] = key

    return sub_fields
