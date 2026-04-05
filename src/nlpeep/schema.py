from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FieldRole(StrEnum):
    QUERY = "query"
    INPUT = "input"
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
            FieldRole.INPUT: "Input",
            FieldRole.RESPONSE: "Response",
            FieldRole.DOCUMENTS: "Documents",
            FieldRole.METRICS: "Metrics",
            FieldRole.TRACE: "Trace",
            FieldRole.GROUND_TRUTH: "Ground Truth",
            FieldRole.ID: "ID",
            FieldRole.METADATA: "Metadata",
            FieldRole.UNMAPPED: "Details",
        }[self]


class FieldArchetype(StrEnum):
    """Data-property-based classification of a field, independent of its role."""

    IDENTIFIER = "identifier"
    FREE_TEXT = "free_text"
    CATEGORY = "category"
    SCORE = "score"
    SCORE_DICT = "score_dict"
    RANKED_LIST = "ranked_list"
    SEQUENCE = "sequence"
    CONVERSATION = "conversation"
    STEPS = "steps"
    FLAT_RECORD = "flat_record"
    NESTED_BLOB = "nested_blob"


class MetricScale(StrEnum):
    """Scale of a numeric metric field."""

    UNIT = "unit"  # [0, 1]
    PERCENT = "percent"  # [0, 100]
    UNKNOWN = "unknown"  # scale not determined


# Split camelCase/PascalCase boundaries for tokenization.
_CAMEL_LOWER_UPPER = re.compile(r"([a-z0-9])([A-Z])")  # "camelCase" -> "camel_Case"
_CAMEL_UPPER_RUN = re.compile(r"([A-Z]+)([A-Z][a-z])")  # "HTMLParser" -> "HTML_Parser"


def _tokenize_name(name: str) -> list[str]:
    """Split a field name into lowercase tokens.

    Handles camelCase, PascalCase, snake_case (including multiple
    underscores), kebab-case, dot.separated, and spaces.  Each token
    is also reduced to a simple singular form so keyword sets only
    need the singular ("metric" matches "metrics").
    """
    name = _CAMEL_LOWER_UPPER.sub(r"\1_\2", name)
    name = _CAMEL_UPPER_RUN.sub(r"\1_\2", name)
    tokens = [t.lower() for t in re.split(r"[_\-.\s]+", name) if t]
    return [_depluralize(t) for t in tokens]


def _depluralize(token: str) -> str:
    """Cheap singular reduction -- covers regular English plurals.

    Intentionally conservative: leaves Latin/Greek endings (-us, -is)
    alone to avoid mangling words like status and analysis.
    """
    if len(token) <= 2:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"  # "entities" -> "entity"
    if token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]  # "classes" -> "class", "searches" -> "search"
    # Skip words that are likely singular despite ending in -s.
    if token.endswith(("us", "is", "ss")):
        return token  # "status", "analysis", "miss"
    if token.endswith("s"):
        return token[:-1]  # "metrics" -> "metric"
    return token


# Token-based role keywords.  A field matches a role when any of its
# tokens appears in the keyword set.  Checked in order; first match wins.
_ROLE_KEYWORDS: list[tuple[FieldRole, frozenset[str]]] = [
    (
        FieldRole.ID,
        frozenset(
            {
                "id",
                "idx",
                "uuid",
                "uid",
                "guid",
                "rowid",
            }
        ),
    ),
    (
        FieldRole.QUERY,
        frozenset(
            {
                "query",
                "question",
                "prompt",
                "search",
                "instruction",
            }
        ),
    ),
    (
        FieldRole.INPUT,
        frozenset(
            {
                "input",
                "source",
                "original",
                "article",
                "sentence",
                "utterance",
                "statement",
                "premise",
                "hypothesis",
                "token",
                "word",
            }
        ),
    ),
    (
        FieldRole.RESPONSE,
        frozenset(
            {
                "response",
                "output",
                "generation",
                "completion",
                "predicted",
                "prediction",
                "reply",
                "generated",
                "summary",
                "abstract",
                "synopsis",
                "translation",
                "translated",
                "answer",
            }
        ),
    ),
    (
        FieldRole.GROUND_TRUTH,
        frozenset(
            {
                "truth",
                "expected",
                "gold",
                "reference",
                "target",
                "actual",
                "label",
                "annotation",
                "annotated",
                "tag",
                "entity",
                "span",
                "slot",
                "sentiment",
                "classification",
                "category",
                "class",
                "topic",
                "intent",
                "emotion",
                "highlight",
            }
        ),
    ),
    (
        FieldRole.DOCUMENTS,
        frozenset(
            {
                "document",
                "doc",
                "context",
                "passage",
                "chunk",
                "retrieved",
                "paragraph",
                "excerpt",
                "snippet",
                "evidence",
            }
        ),
    ),
    (
        FieldRole.METRICS,
        frozenset(
            {
                "metric",
                "score",
                "evaluation",
                "eval",
            }
        ),
    ),
    (
        FieldRole.TRACE,
        frozenset(
            {
                "trace",
                "step",
                "trajectory",
                "history",
                "chain",
                "action",
            }
        ),
    ),
    (
        FieldRole.METADATA,
        frozenset(
            {
                "metadata",
                "meta",
                "config",
                "param",
                "setting",
                "extra",
                "option",
                "hyperparam",
            }
        ),
    ),
]

# Very short or ambiguous tokens that only match as the full field name.
_EXACT_KEYWORDS: dict[str, FieldRole] = {
    "q": FieldRole.QUERY,
    "text": FieldRole.INPUT,
    "document": FieldRole.INPUT,
    "result": FieldRole.RESPONSE,
    "correct": FieldRole.GROUND_TRUTH,
    "info": FieldRole.METADATA,
}

# Maximum recursion depth for nested field detection
_MAX_NEST_DEPTH = 4


@dataclass
class FieldMapping:
    json_path: str
    role: FieldRole
    display_name: str = ""
    sub_fields: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    archetype: FieldArchetype | None = None
    metric_scale: MetricScale | None = None

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

    def resolve_all(self, record_data: dict[str, Any], role: FieldRole) -> dict[str, Any]:
        """Return {json_path: value} for ALL mappings with the given role."""
        result: dict[str, Any] = {}
        for m in self.mappings:
            if m.role == role:
                val = _resolve_path(record_data, m.json_path)
                if val is not None:
                    result[m.json_path] = val
        return result

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
        # Build set of top-level keys that are fully or partially consumed
        consumed_top_keys: set[str] = set()
        for p in mapped_paths:
            consumed_top_keys.add(p.split(".")[0])

        result: dict[str, Any] = {}
        for key, val in record_data.items():
            if key in mapped_paths:
                # Exact top-level match -- skip entirely
                continue
            if key in consumed_top_keys:
                # Some nested path under this key is mapped.
                # Include only the parts that are not mapped.
                remaining = _unmapped_subtree(key, val, mapped_paths)
                if remaining is not None:
                    result[key] = remaining
            else:
                result[key] = val
        return result

    def archetype_for_path(self, path: str) -> FieldArchetype | None:
        """Return the archetype for a given json_path, or None."""
        for m in self.mappings:
            if m.json_path == path:
                return m.archetype
        return None

    def label_path(self) -> str | None:
        for role in (FieldRole.QUERY, FieldRole.INPUT, FieldRole.ID):
            m = self.get_mapping(role)
            if m:
                return m.json_path
        return None

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"mapping": {}, "display": {}}
        # Group mappings by role to support multi-field roles
        role_paths: dict[str, list[str]] = {}
        role_sub_fields: dict[str, dict[str, str]] = {}
        role_display: dict[str, str] = {}
        for m in self.mappings:
            if m.role == FieldRole.UNMAPPED:
                continue
            role_paths.setdefault(m.role.value, []).append(m.json_path)
            if m.sub_fields:
                role_sub_fields[m.role.value] = m.sub_fields
            if m.display_name != m.role.display_name:
                role_display[m.role.value] = m.display_name
        for role_val, paths in role_paths.items():
            config["mapping"][role_val] = paths[0] if len(paths) == 1 else paths
            if role_val in role_sub_fields:
                config["mapping"][f"{role_val}_fields"] = role_sub_fields[role_val]
            if role_val in role_display:
                config["display"][role_val] = role_display[role_val]
        # Persist metric scales
        scales: dict[str, str] = {}
        for m in self.mappings:
            if m.metric_scale is not None:
                scales[m.json_path] = m.metric_scale.value
        if scales:
            config["metric_scales"] = scales
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SchemaMapping:
        mapping_section = config.get("mapping", {})
        display_section = config.get("display", {})
        scales_section = config.get("metric_scales", {})
        mappings: list[FieldMapping] = []

        for role in FieldRole:
            if role == FieldRole.UNMAPPED:
                continue
            path = mapping_section.get(role.value)
            if not path:
                continue
            sub_fields = mapping_section.get(f"{role.value}_fields", {})
            display_name = display_section.get(role.value, role.display_name)
            # Support both single string and list of strings
            paths = path if isinstance(path, list) else [path]
            for p in paths:
                if isinstance(p, str):
                    scale_str = scales_section.get(p)
                    try:
                        metric_scale = MetricScale(scale_str) if scale_str else None
                    except ValueError:
                        metric_scale = None
                    mappings.append(
                        FieldMapping(
                            json_path=p,
                            role=role,
                            display_name=display_name,
                            sub_fields=sub_fields if isinstance(sub_fields, dict) else {},
                            confidence=1.0,
                            metric_scale=metric_scale,
                        )
                    )

        return cls(mappings=mappings)

    @classmethod
    def auto_detect(cls, records: list[Any]) -> SchemaMapping:
        if not records:
            return cls()

        # Collect all fields (including nested) across sample records.
        # Each entry maps a dot-path to its occurrence count and sample values.
        path_counts: dict[str, int] = {}
        path_samples: dict[str, list[Any]] = {}
        path_keys: dict[str, str] = {}  # dot_path -> original key name
        total = len(records)

        for record in records:
            data = record.data if hasattr(record, "data") else record
            if not isinstance(data, dict):
                continue
            flat, orig_keys = _flatten_record(data)
            for dot_path, val in flat.items():
                path_counts[dot_path] = path_counts.get(dot_path, 0) + 1
                if dot_path not in path_samples:
                    path_samples[dot_path] = []
                if len(path_samples[dot_path]) < 5:
                    path_samples[dot_path].append(val)
                if dot_path not in path_keys:
                    path_keys[dot_path] = orig_keys.get(dot_path, dot_path)

        mappings: list[FieldMapping] = []
        claimed_roles: set[FieldRole] = set()

        # Pass 1: Token-based keyword matching on field names.
        # Sort paths so shallower ones come first (fewer dots = shallower).
        sorted_paths = sorted(path_counts, key=lambda p: p.count("."))
        for dot_path in sorted_paths:
            # Build candidate names from the original key (preserves
            # Phoenix-style dotted keys like "input.value") and the
            # flattened leaf name.
            original_key = path_keys.get(dot_path, dot_path.rsplit(".", 1)[-1])
            candidates = [original_key]
            if "." in original_key:
                candidates.extend(original_key.split("."))

            # Tokenize all candidate names for keyword matching.
            tokens: set[str] = set()
            for c in candidates:
                tokens.update(_tokenize_name(c))

            samples = path_samples.get(dot_path, [])
            is_nested = "." in dot_path

            # Check exact-only keywords first (very short/ambiguous tokens).
            exact_lower = original_key.lower()
            exact_role = _EXACT_KEYWORDS.get(exact_lower)

            for role, keywords in _ROLE_KEYWORDS:
                if role in claimed_roles:
                    continue
                # Match if any token hits the keyword set, or the exact
                # field name matched this role via _EXACT_KEYWORDS.
                if not (tokens & keywords or exact_role == role):
                    continue
                # Depth guard: generic INPUT tokens ("text", "source")
                # are ambiguous when nested (e.g. "answers.text").
                if is_nested and role == FieldRole.INPUT:
                    continue
                # Type guard: input/output roles need text-like samples
                # (string or list-of-strings), not bare ints.  Ground truth
                # is exempt -- labels can be ints or int-lists.
                if (
                    role in (FieldRole.QUERY, FieldRole.INPUT, FieldRole.RESPONSE)
                    and samples
                    and not any(
                        isinstance(s, str) or (isinstance(s, list) and s and isinstance(s[0], str))
                        for s in samples
                    )
                ):
                    continue
                # Presence guard: IDs should appear in most records.
                if role == FieldRole.ID and path_counts[dot_path] / total < 0.5:
                    continue

                presence = path_counts[dot_path] / total
                confidence = 0.9 * presence
                mapping = FieldMapping(
                    json_path=dot_path,
                    role=role,
                    confidence=confidence,
                )
                if role == FieldRole.DOCUMENTS:
                    mapping.sub_fields = _detect_doc_subfields(samples)
                mappings.append(mapping)
                claimed_roles.add(role)
                break

        # Pass 2: Structural analysis for unclaimed roles
        for dot_path in sorted_paths:
            if any(m.json_path == dot_path for m in mappings):
                continue
            samples = path_samples.get(dot_path, [])
            leaf = dot_path.rsplit(".", 1)[-1]
            role = _structural_role(leaf, samples)
            if role != FieldRole.UNMAPPED and role not in claimed_roles:
                presence = path_counts[dot_path] / total
                mapping = FieldMapping(
                    json_path=dot_path,
                    role=role,
                    confidence=0.5 * presence,
                )
                if role == FieldRole.DOCUMENTS:
                    mapping.sub_fields = _detect_doc_subfields(samples)
                mappings.append(mapping)
                claimed_roles.add(role)

        # Pass 3: Scan remaining unmapped paths for score-like floats.
        # These get assigned to METRICS (allowing multiple fields per role).
        mapped_paths = {m.json_path for m in mappings}
        for dot_path in sorted_paths:
            if dot_path in mapped_paths:
                continue
            samples = path_samples.get(dot_path, [])
            leaf = dot_path.rsplit(".", 1)[-1]
            scale = _is_score_like_field(leaf, samples)
            if scale is not None:
                presence = path_counts[dot_path] / total
                mappings.append(
                    FieldMapping(
                        json_path=dot_path,
                        role=FieldRole.METRICS,
                        confidence=0.6 * presence,
                        metric_scale=scale,
                    )
                )

        # Pass 4: Classify archetypes for all discovered paths.
        # This provides rendering hints even for unmapped fields.
        mapped_paths = {m.json_path for m in mappings}
        for m in mappings:
            m.archetype = _classify_archetype(m.json_path, path_samples.get(m.json_path, []), total)
        # Also assign archetypes to remaining top-level fields that
        # were not mapped (they will render via archetype in Details tab).
        for dot_path in sorted_paths:
            if dot_path in mapped_paths:
                continue
            arch = _classify_archetype(dot_path, path_samples.get(dot_path, []), total)
            # Only store archetype info for leaf fields that are not
            # parents of already-mapped deeper paths, to avoid
            # duplicating container dicts.
            mappings.append(
                FieldMapping(
                    json_path=dot_path,
                    role=FieldRole.UNMAPPED,
                    confidence=0.0,
                    archetype=arch,
                )
            )

        return cls(mappings=mappings)


def _flatten_record(
    data: dict[str, Any],
    prefix: str = "",
    depth: int = 0,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Flatten nested dicts into dot-paths, stopping at lists and the depth cap.

    Returns two dicts:
      - {dot_path: leaf_value}
      - {dot_path: original_key} so callers can match against the real
        key name (important for formats with dots in key names).
    """
    paths: dict[str, Any] = {}
    keys: dict[str, str] = {}
    for key, val in data.items():
        dot_path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and depth < _MAX_NEST_DEPTH:
            sub_paths, sub_keys = _flatten_record(val, dot_path, depth + 1)
            paths.update(sub_paths)
            keys.update(sub_keys)
        else:
            paths[dot_path] = val
            keys[dot_path] = key
    return paths, keys


def _classify_archetype(path: str, samples: list[Any], total_records: int) -> FieldArchetype | None:
    """Classify a field based on its sampled data properties."""
    if not samples:
        return None

    # -- identifier: unique per record, short string or int --
    if all(isinstance(s, (str, int)) for s in samples):
        if all(isinstance(s, int) for s in samples):
            unique = len(set(samples))
            if unique == len(samples):
                return FieldArchetype.IDENTIFIER
        elif all(isinstance(s, str) for s in samples):
            unique = len(set(samples))
            avg_len = sum(len(s) for s in samples) / len(samples) if samples else 0
            if unique == len(samples) and avg_len < 100:
                return FieldArchetype.IDENTIFIER

    # -- score: float in bounded range [0, 1] or [0, 100] for known metric names --
    numeric = [s for s in samples if isinstance(s, (int, float)) and not isinstance(s, bool)]
    if numeric and len(numeric) == len(samples):
        if all(0 <= v <= 1 for v in numeric) and any(isinstance(v, float) for v in numeric):
            return FieldArchetype.SCORE
        if all(0 <= v <= 100 for v in numeric) and any(v > 1 for v in numeric):
            leaf = path.rsplit(".", 1)[-1]
            key_lower = leaf.lower().replace("_", "").replace("-", "")
            if any(sub in key_lower for sub in _SCORE_SUBSTRINGS):
                return FieldArchetype.SCORE

    # -- score_dict: dict where all values are numeric --
    if all(isinstance(s, dict) for s in samples) and all(
        s and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in s.values())
        for s in samples
    ):
        return FieldArchetype.SCORE_DICT

    # -- conversation: list of dicts with role + content pattern --
    if all(isinstance(s, list) for s in samples):
        flat_items = [item for s in samples for item in s if isinstance(item, dict)]
        if flat_items and all("role" in item and "content" in item for item in flat_items[:10]):
            return FieldArchetype.CONVERSATION

    # -- ranked_list: list of dicts containing a text field + a numeric field --
    if all(isinstance(s, list) for s in samples):
        flat_items = [item for s in samples for item in s if isinstance(item, dict)]
        if flat_items and _looks_like_docs(flat_items):
            return FieldArchetype.RANKED_LIST

    # -- steps: list of dicts with action/tool-like keys --
    if all(isinstance(s, list) for s in samples):
        flat_items = [item for s in samples for item in s if isinstance(item, dict)]
        if flat_items and _looks_like_trace(flat_items):
            return FieldArchetype.STEPS

    # -- sequence: list of short strings (tokens) --
    if all(isinstance(s, list) for s in samples):
        flat_items = [item for s in samples for item in s]
        if flat_items and all(isinstance(item, str) for item in flat_items):
            avg_len = sum(len(item) for item in flat_items) / len(flat_items)
            if avg_len < 30:
                return FieldArchetype.SEQUENCE

    # -- String-based archetypes: free_text vs category --
    str_samples = [s for s in samples if isinstance(s, str)]
    if str_samples and len(str_samples) == len(samples):
        avg_len = sum(len(s) for s in str_samples) / len(str_samples)
        lengths = [len(s) for s in str_samples]
        length_variance = (
            sum((ln - avg_len) ** 2 for ln in lengths) / len(lengths) if len(lengths) > 1 else 0.0
        )
        unique = len(set(str_samples))
        cardinality_ratio = unique / len(str_samples) if str_samples else 1.0

        if avg_len > 50:
            return FieldArchetype.FREE_TEXT
        # Category: low cardinality, short strings, low length variance
        if cardinality_ratio < 0.8 and avg_len < 50 and length_variance < 200:
            return FieldArchetype.CATEGORY
        # Small sample with short strings -- use length variance as tiebreaker
        if avg_len < 30 and length_variance < 50:
            return FieldArchetype.CATEGORY
        if avg_len > 20:
            return FieldArchetype.FREE_TEXT

    # -- flat_record: dict of scalar values --
    if all(isinstance(s, dict) for s in samples) and all(
        all(isinstance(v, (str, int, float, bool, type(None))) for v in s.values())
        for s in samples
        if s
    ):
        return FieldArchetype.FLAT_RECORD

    # -- nested_blob: anything else complex --
    if any(isinstance(s, (dict, list)) for s in samples):
        return FieldArchetype.NESTED_BLOB

    return None


def _unmapped_subtree(prefix: str, val: Any, mapped_paths: set[str]) -> Any:
    """Return the portion of *val* whose paths are not in *mapped_paths*.

    For a dict, filters out keys whose dot-path is mapped, recursing one level.
    Returns None if everything is consumed.
    """
    if not isinstance(val, dict):
        return val
    remaining: dict[str, Any] = {}
    for k, v in val.items():
        full = f"{prefix}.{k}"
        if full in mapped_paths:
            continue
        # Check if any mapped path is beneath this key
        child_mapped = any(p.startswith(full + ".") for p in mapped_paths)
        if child_mapped:
            sub = _unmapped_subtree(full, v, mapped_paths)
            if sub is not None:
                remaining[k] = sub
        else:
            remaining[k] = v
    return remaining if remaining else None


def _resolve_path(data: Any, path: str) -> Any:
    """Resolve a dot-path against nested data, handling dotted key names.

    Uses greedy matching: when a single segment doesn't match a dict key,
    tries progressively longer dot-joined segments.  This handles formats
    like Phoenix OpenInference where keys contain literal dots
    (e.g. ``attributes["input.value"]``).
    """
    parts = path.split(".")
    current = data
    i = 0
    while i < len(parts):
        if isinstance(current, dict):
            # Try progressively longer key segments (greedy)
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
    if all(isinstance(s, dict) for s in samples) and all(
        all(isinstance(v, int | float) for v in s.values()) for s in samples if s
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


_SCORE_SUBSTRINGS = {
    "score",
    "relevance",
    "similarity",
    "confidence",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "bleu",
    "rouge",
    "coherence",
    "faithfulness",
    "relevancy",
    "ndcg",
    "mrr",
    "distance",
}


def _is_score_like_field(key: str, samples: list[Any]) -> MetricScale | None:
    """Check if a key name contains score-like substrings and infer the scale."""
    key_lower = key.lower().replace("_", "").replace("-", "")
    if not any(sub in key_lower for sub in _SCORE_SUBSTRINGS):
        return None
    numeric_samples = [
        s for s in samples if isinstance(s, (int, float)) and not isinstance(s, bool)
    ]
    if not numeric_samples:
        return None
    if all(0 <= v <= 1 for v in numeric_samples):
        return MetricScale.UNIT
    if all(0 <= v <= 100 for v in numeric_samples) and any(v > 1 for v in numeric_samples):
        return MetricScale.PERCENT
    return None


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
