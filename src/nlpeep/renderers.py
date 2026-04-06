from __future__ import annotations

import json
from enum import Enum
from typing import Any

from rich.markup import escape
from rich.syntax import Syntax
from textual.widget import Widget
from textual.widgets import Collapsible, DataTable, Markdown, Static, Tree

from nlpeep.schema import FieldArchetype, FieldRole, MetricScale


class ValueType(Enum):
    SHORT_TEXT = "short_text"
    LONG_TEXT = "long_text"
    MARKDOWN = "markdown"
    SCORE = "score"
    METRIC_DICT = "metric_dict"
    DOC_LIST = "doc_list"
    CHAT_MESSAGES = "chat_messages"
    STEP_LIST = "step_list"
    FLAT_DICT = "flat_dict"
    NESTED_DICT = "nested_dict"
    SIMPLE_LIST = "simple_list"
    TAGGED_SEQUENCE = "tagged_sequence"
    TABLE = "table"
    JSON_RAW = "json_raw"


# Field names that suggest a value in [0,1] is a score
_SCORE_NAMES = {
    "score",
    "relevance",
    "similarity",
    "confidence",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "ndcg",
    "mrr",
    "map",
    "bleu",
    "rouge",
    "faithfulness",
    "relevancy",
    "coherence",
    "distance",
}


# Mapping from archetype to the best ValueType for rendering unmapped fields
_ARCHETYPE_TO_VALUE_TYPE: dict[FieldArchetype, ValueType] = {
    FieldArchetype.IDENTIFIER: ValueType.SHORT_TEXT,
    FieldArchetype.FREE_TEXT: ValueType.LONG_TEXT,
    FieldArchetype.CATEGORY: ValueType.SHORT_TEXT,
    FieldArchetype.SCORE: ValueType.SCORE,
    FieldArchetype.SCORE_DICT: ValueType.METRIC_DICT,
    FieldArchetype.RANKED_LIST: ValueType.DOC_LIST,
    FieldArchetype.TAGGED_SEQUENCE: ValueType.TAGGED_SEQUENCE,
    FieldArchetype.SEQUENCE: ValueType.SIMPLE_LIST,
    FieldArchetype.CONVERSATION: ValueType.CHAT_MESSAGES,
    FieldArchetype.STEPS: ValueType.STEP_LIST,
    FieldArchetype.FLAT_RECORD: ValueType.FLAT_DICT,
    FieldArchetype.NESTED_BLOB: ValueType.NESTED_DICT,
}


def classify_value(
    value: Any,
    field_name: str = "",
    role: FieldRole = FieldRole.UNMAPPED,
    archetype: FieldArchetype | None = None,
) -> ValueType:
    """Three-pass classification: role-based first, archetype second, then structural."""
    # Pass 1: Role-based
    if role in (FieldRole.QUERY, FieldRole.INPUT):
        if isinstance(value, str):
            return (
                ValueType.SHORT_TEXT
                if len(value) < 200 and "\n" not in value
                else ValueType.LONG_TEXT
            )
    elif role == FieldRole.RESPONSE:
        if isinstance(value, str):
            return ValueType.MARKDOWN
    elif role == FieldRole.GROUND_TRUTH:
        if isinstance(value, str):
            return ValueType.MARKDOWN
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return ValueType.SHORT_TEXT
        if isinstance(value, list):
            return ValueType.SIMPLE_LIST
    elif role == FieldRole.DOCUMENTS:
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                return ValueType.DOC_LIST
            return ValueType.SIMPLE_LIST
    elif role == FieldRole.METRICS:
        if isinstance(value, dict):
            return ValueType.METRIC_DICT
    elif (
        role == FieldRole.TRACE and isinstance(value, list) and value and isinstance(value[0], dict)
    ):
        keys = set(value[0].keys())
        if {"role", "content"} <= keys:
            return ValueType.CHAT_MESSAGES
        return ValueType.STEP_LIST

    # Pass 2: Archetype-based (only when no role dictated a type above)
    if archetype is not None and role == FieldRole.UNMAPPED:
        archetype_type = _ARCHETYPE_TO_VALUE_TYPE.get(archetype)
        if archetype_type is not None:
            # For archetypes that map to list-based types, verify the value is actually a list
            if archetype_type in (
                ValueType.DOC_LIST,
                ValueType.CHAT_MESSAGES,
                ValueType.STEP_LIST,
                ValueType.SIMPLE_LIST,
            ) and not isinstance(value, list):
                pass  # fall through to structural
            else:
                return archetype_type

    # Pass 3: Structural
    if isinstance(value, str):
        if len(value) < 200 and "\n" not in value:
            return ValueType.SHORT_TEXT
        if _looks_like_markdown(value):
            return ValueType.MARKDOWN
        return ValueType.LONG_TEXT

    if isinstance(value, bool):
        return ValueType.SHORT_TEXT

    if isinstance(value, (int, float)):
        name_lower = field_name.lower().replace("_", "").replace("-", "")
        if any(s in name_lower for s in _SCORE_NAMES) and 0 <= value <= 1:
            return ValueType.SCORE
        return ValueType.SHORT_TEXT

    if isinstance(value, list):
        if not value:
            return ValueType.SHORT_TEXT
        if all(isinstance(item, str) for item in value):
            return ValueType.SIMPLE_LIST
        if all(isinstance(item, dict) for item in value):
            first = value[0]
            keys = set(first.keys())
            if {"role", "content"} <= keys:
                return ValueType.CHAT_MESSAGES
            if any(k.lower() in {"step", "action", "tool", "type", "function"} for k in keys):
                return ValueType.STEP_LIST
            return ValueType.TABLE
        return ValueType.JSON_RAW

    if isinstance(value, dict):
        if not value:
            return ValueType.SHORT_TEXT
        # Tagged sequence: aligned "tokens" and "tags" lists
        if "tokens" in value and "tags" in value:
            tokens, tags = value["tokens"], value["tags"]
            if (
                isinstance(tokens, list)
                and isinstance(tags, list)
                and len(tokens) == len(tags)
                and tokens
            ):
                return ValueType.TAGGED_SEQUENCE
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()):
            return ValueType.METRIC_DICT
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in value.values()):
            return ValueType.FLAT_DICT
        return ValueType.NESTED_DICT

    return ValueType.JSON_RAW


def _looks_like_markdown(text: str) -> bool:
    indicators = ["# ", "## ", "**", "```", "- ", "1. ", "* ", "| "]
    return any(indicator in text for indicator in indicators)


def _score_color(value: float, scale: MetricScale | None) -> str | None:
    """Return a Rich color token for a metric value, or None if scale is unknown."""
    if scale is None or scale == MetricScale.UNKNOWN:
        return None
    norm = value / 100.0 if scale == MetricScale.PERCENT else value
    if not (0 <= norm <= 1):
        return None
    if norm >= 0.8:
        return "$success"
    if norm >= 0.5:
        return "$warning"
    return "$error"


def render_value(
    value: Any,
    value_type: ValueType,
    field_name: str = "",
    sub_fields: dict[str, str] | None = None,
    metric_scale: MetricScale | None = None,
) -> Widget:
    """Return a Textual Widget for the given value and classification."""
    try:
        return _RENDERERS[value_type](value, field_name, sub_fields or {}, metric_scale)
    except Exception:
        return _render_json_raw(value, field_name, {}, None)


def _render_short_text(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    return Static(str(value), classes="field-short-text")


def _render_long_text(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    return Static(str(value), classes="field-long-text")


def _render_markdown(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    return Markdown(str(value), classes="field-markdown")


def _render_score(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    v = float(value)
    formatted = f"{v:.3f}"
    color = _score_color(v, metric_scale)
    if color:
        text = f"[bold]{escape(field_name)}[/bold]  [{color}]{formatted}[/{color}]"
    else:
        text = f"[bold]{escape(field_name)}[/bold]  {formatted}"
    return Static(text, classes="field-metrics")


def _render_metric_dict(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    numeric = {
        k: float(v)
        for k, v in value.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not numeric:
        return _render_json_raw(value, field_name, sub_fields, None)
    label_width = max(len(k) for k in numeric)
    lines: list[str] = []
    for k, v in numeric.items():
        padded = k.rjust(label_width)
        formatted = f"{v:.3f}"
        # Per-key scale from sub_fields, fall back to overall metric_scale
        key_scale_str = sub_fields.get(k)
        if key_scale_str:
            try:
                key_scale = MetricScale(key_scale_str)
            except ValueError:
                key_scale = metric_scale
        else:
            key_scale = metric_scale
        color = _score_color(v, key_scale)
        if color:
            lines.append(f"[bold]{escape(padded)}[/bold]  [{color}]{formatted}[/{color}]")
        else:
            lines.append(f"[bold]{escape(padded)}[/bold]  {formatted}")
    return Static("\n".join(lines), classes="field-metrics")


def _render_doc_list(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    from textual.containers import Vertical

    from nlpeep.widgets.doc_card import DocCard

    cards = []
    for i, doc in enumerate(value):
        if isinstance(doc, dict):
            cards.append(DocCard(doc=doc, rank=i + 1, sub_fields=sub_fields))
        else:
            cards.append(Static(str(doc)))
    return Vertical(*cards, classes="field-docs")


def _render_chat_messages(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    from textual.containers import Vertical

    widgets: list[Widget] = []
    for msg in value:
        if not isinstance(msg, dict):
            widgets.append(Static(str(msg)))
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle structured content (e.g., tool results)
            content = json.dumps(content, indent=2)
        role_colors = {
            "system": "$warning",
            "user": "$accent",
            "assistant": "$success",
            "tool": "$secondary",
            "function": "$secondary",
        }
        color = role_colors.get(role, "white")
        header = f"[bold {color}]{escape(role)}[/bold {color}]"

        # Show tool call info if present
        extra = ""
        if "tool_calls" in msg:
            calls = msg["tool_calls"]
            if isinstance(calls, list):
                names = [
                    c.get("function", {}).get("name", "?") if isinstance(c, dict) else "?"
                    for c in calls
                ]
                extra = f"\n[dim]Tool calls: {', '.join(names)}[/dim]"

        text = f"{header}\n{escape(str(content))}{extra}"
        widgets.append(Static(text, classes=f"chat-message chat-{role}"))

    return Vertical(*widgets, classes="field-chat")


def _render_step_list(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    from textual.containers import Vertical

    widgets: list[Widget] = []
    for i, step in enumerate(value):
        if not isinstance(step, dict):
            widgets.append(Static(str(step)))
            continue
        # Build a label from available fields
        label_parts = []
        for key in ("step", "action", "type", "tool", "function", "name"):
            if key in step:
                label_parts.append(f"{key}={step[key]}")
        label = ", ".join(label_parts) if label_parts else f"Step {i + 1}"

        # Render the step content
        detail_lines = []
        for k, v in step.items():
            if k in ("step", "action", "type"):
                continue
            if isinstance(v, (dict, list)):
                detail_lines.append(f"[bold]{escape(k)}[/bold]: {escape(json.dumps(v, indent=2))}")
            else:
                detail_lines.append(f"[bold]{escape(k)}[/bold]: {escape(str(v))}")

        content = Static("\n".join(detail_lines), classes="step-content")
        widgets.append(Collapsible(content, title=label, collapsed=i > 0))

    return Vertical(*widgets, classes="field-steps")


def _render_flat_dict(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    table = DataTable(classes="field-flat-dict")
    table.add_columns("Field", "Value")
    for k, v in value.items():
        table.add_row(str(k), str(v))
    return table


def _render_nested_dict(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    tree: Tree[str] = Tree(field_name or "Object", classes="field-tree")
    _build_tree(tree.root, value)
    tree.root.expand_all()
    return tree


def _build_tree(node: Any, data: Any) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                child = node.add(f"[bold]{escape(str(k))}[/bold]")
                _build_tree(child, v)
            else:
                node.add_leaf(f"[bold]{escape(str(k))}[/bold]: {escape(str(v))}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                child = node.add(f"[dim]\\[{i}][/dim]")
                _build_tree(child, item)
            else:
                node.add_leaf(f"[dim]\\[{i}][/dim] {escape(str(item))}")
    else:
        node.add_leaf(escape(str(data)))


def _render_simple_list(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    items = [f"  [dim]{BULLET}[/dim] {escape(str(item))}" for item in value]
    return Static("\n".join(items), classes="field-list")


BULLET = "\u2022"


def _render_table(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    if not value:
        return Static("(empty)", classes="field-table")

    table = DataTable(classes="field-table")
    # Collect all keys across all rows
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in value:
        if isinstance(row, dict):
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    table.add_columns(*all_keys)
    for row in value:
        if isinstance(row, dict):
            cells = []
            for k in all_keys:
                v = row.get(k, "")
                if isinstance(v, (dict, list)):
                    cells.append(json.dumps(v, ensure_ascii=False)[:100])
                else:
                    cells.append(str(v))
            table.add_row(*cells)

    return table


def _render_json_raw(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    syntax = Syntax(text, "json", theme="monokai", line_numbers=False)
    return Static(syntax, classes="field-json")


# -- Tagged-sequence (NER) renderer ------------------------------------------

_NER_COLORS: dict[str, str] = {
    "PER": "cyan",
    "PERSON": "cyan",
    "ORG": "green",
    "ORGANIZATION": "green",
    "LOC": "yellow",
    "LOCATION": "yellow",
    "GPE": "yellow",
    "MISC": "magenta",
    "DATE": "blue",
    "TIME": "blue",
    "MONEY": "bright_green",
    "PERCENT": "bright_green",
    "FAC": "bright_magenta",
    "EVENT": "bright_cyan",
    "PRODUCT": "bright_yellow",
    "NORP": "bright_green",
    "QUANTITY": "bright_green",
    "ORDINAL": "bright_blue",
    "CARDINAL": "bright_blue",
}


def _parse_bio_tag(tag: str) -> tuple[str, str]:
    """Parse a BIO/IOB2 tag into (prefix, entity_type).

    Returns ("O", "O") for outside tags, or (prefix, type) for entities.
    """
    if tag in ("O", "o"):
        return ("O", "O")
    if "-" in tag:
        prefix, entity = tag.split("-", 1)
        return (prefix, entity)
    return ("B", tag)


def _render_tagged_sequence(
    value: Any, field_name: str, sub_fields: dict[str, str], metric_scale: MetricScale | None
) -> Widget:
    """Render aligned tokens and tags as an annotated sequence."""
    if not isinstance(value, dict) or "tokens" not in value or "tags" not in value:
        return _render_json_raw(value, field_name, sub_fields, metric_scale)

    tokens = value["tokens"]
    tags = value["tags"]

    if not isinstance(tokens, list) or not isinstance(tags, list):
        return _render_json_raw(value, field_name, sub_fields, metric_scale)

    if len(tokens) != len(tags):
        return _render_json_raw(value, field_name, sub_fields, metric_scale)

    if not tokens:
        return Static("(empty sequence)", classes="field-tagged-seq")

    if all(isinstance(t, str) for t in tags):
        return _render_bio_tags(tokens, tags)
    if all(isinstance(t, int) for t in tags):
        return _render_int_tags(tokens, tags)
    return _render_json_raw(value, field_name, sub_fields, metric_scale)


def _render_bio_tags(tokens: list[str], tags: list[str]) -> Widget:
    """Render tokens with BIO/IOB2 string tags, grouping entity spans."""
    parts: list[str] = []
    i = 0
    while i < len(tokens):
        prefix, etype = _parse_bio_tag(tags[i])
        if prefix == "O":
            parts.append(escape(tokens[i]))
            i += 1
        else:
            # Collect the full entity span (B-X followed by I-X)
            entity_tokens = [tokens[i]]
            j = i + 1
            while j < len(tokens):
                p2, e2 = _parse_bio_tag(tags[j])
                if p2 == "I" and e2 == etype:
                    entity_tokens.append(tokens[j])
                    j += 1
                else:
                    break
            entity_text = " ".join(escape(t) for t in entity_tokens)
            color = _NER_COLORS.get(etype.upper(), "bright_white")
            parts.append(f"[bold {color}]{entity_text}[/bold {color}][dim]({etype})[/dim]")
            i = j

    return Static(" ".join(parts), classes="field-tagged-seq")


def _render_int_tags(tokens: list[str], tags: list[int]) -> Widget:
    """Render tokens with integer tags in a two-line aligned format."""
    widths = [max(len(tok), len(str(tag))) + 2 for tok, tag in zip(tokens, tags, strict=False)]

    # Token line: pad before escaping so display width stays correct
    token_parts = [escape(tok.ljust(w)) for tok, w in zip(tokens, widths, strict=False)]

    # Tag line: integer tags with non-zero highlighted
    tag_parts: list[str] = []
    for tag, w in zip(tags, widths, strict=False):
        tag_str = str(tag).ljust(w)
        if tag == 0:
            tag_parts.append(f"[dim]{tag_str}[/dim]")
        else:
            tag_parts.append(f"[bold bright_cyan]{tag_str}[/bold bright_cyan]")

    text = "".join(token_parts).rstrip() + "\n" + "".join(tag_parts).rstrip()
    return Static(text, classes="field-tagged-seq")


_RENDERERS = {
    ValueType.SHORT_TEXT: _render_short_text,
    ValueType.LONG_TEXT: _render_long_text,
    ValueType.MARKDOWN: _render_markdown,
    ValueType.SCORE: _render_score,
    ValueType.METRIC_DICT: _render_metric_dict,
    ValueType.DOC_LIST: _render_doc_list,
    ValueType.CHAT_MESSAGES: _render_chat_messages,
    ValueType.STEP_LIST: _render_step_list,
    ValueType.FLAT_DICT: _render_flat_dict,
    ValueType.NESTED_DICT: _render_nested_dict,
    ValueType.TAGGED_SEQUENCE: _render_tagged_sequence,
    ValueType.SIMPLE_LIST: _render_simple_list,
    ValueType.TABLE: _render_table,
    ValueType.JSON_RAW: _render_json_raw,
}
