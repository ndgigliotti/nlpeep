from __future__ import annotations

import json
from enum import Enum
from typing import Any

from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.widget import Widget
from textual.widgets import DataTable, Markdown, Static, Tree, Collapsible

from rag_viewer.schema import FieldRole


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
    TABLE = "table"
    JSON_RAW = "json_raw"


# Field names that suggest a value in [0,1] is a score
_SCORE_NAMES = {
    "score", "relevance", "similarity", "confidence", "precision",
    "recall", "f1", "accuracy", "ndcg", "mrr", "map", "bleu", "rouge",
    "faithfulness", "relevancy", "coherence", "distance",
}


def classify_value(value: Any, field_name: str = "", role: FieldRole = FieldRole.UNMAPPED) -> ValueType:
    """Two-pass classification: role-based first, then structural."""
    # Pass 1: Role-based
    if role == FieldRole.QUERY:
        if isinstance(value, str):
            return ValueType.SHORT_TEXT if len(value) < 200 and "\n" not in value else ValueType.LONG_TEXT
    elif role == FieldRole.RESPONSE:
        if isinstance(value, str):
            return ValueType.MARKDOWN
    elif role == FieldRole.GROUND_TRUTH:
        if isinstance(value, str):
            return ValueType.MARKDOWN
    elif role == FieldRole.DOCUMENTS:
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                return ValueType.DOC_LIST
            return ValueType.SIMPLE_LIST
    elif role == FieldRole.METRICS:
        if isinstance(value, dict):
            return ValueType.METRIC_DICT
    elif role == FieldRole.TRACE:
        if isinstance(value, list) and value:
            if isinstance(value[0], dict):
                keys = set(value[0].keys())
                if {"role", "content"} <= keys:
                    return ValueType.CHAT_MESSAGES
                return ValueType.STEP_LIST

    # Pass 2: Structural
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
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()):
            return ValueType.METRIC_DICT
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in value.values()):
            return ValueType.FLAT_DICT
        return ValueType.NESTED_DICT

    return ValueType.JSON_RAW


def _looks_like_markdown(text: str) -> bool:
    indicators = ["# ", "## ", "**", "```", "- ", "1. ", "* ", "| "]
    return any(indicator in text for indicator in indicators)


def render_value(value: Any, value_type: ValueType, field_name: str = "", sub_fields: dict[str, str] | None = None) -> Widget:
    """Return a Textual Widget for the given value and classification."""
    try:
        return _RENDERERS[value_type](value, field_name, sub_fields or {})
    except Exception:
        return _render_json_raw(value, field_name, {})


def _render_short_text(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    return Static(str(value), classes="field-short-text")


def _render_long_text(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    return Static(str(value), classes="field-long-text")


def _render_markdown(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    return Markdown(str(value), classes="field-markdown")


def _render_score(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    from rag_viewer.widgets.score_bar import ScoreBar
    return ScoreBar(value=float(value), label=field_name)


def _render_metric_dict(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    from rag_viewer.widgets.score_bar import ScoreBar
    from textual.containers import Vertical
    bars = []
    for k, v in value.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            bars.append(ScoreBar(value=float(v), label=k))
    if not bars:
        return _render_json_raw(value, field_name, sub_fields)
    return Vertical(*bars, classes="field-metrics")


def _render_doc_list(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    from rag_viewer.widgets.doc_card import DocCard
    from textual.containers import Vertical
    cards = []
    for i, doc in enumerate(value):
        if isinstance(doc, dict):
            cards.append(DocCard(doc=doc, rank=i + 1, sub_fields=sub_fields))
        else:
            cards.append(Static(str(doc)))
    return Vertical(*cards, classes="field-docs")


def _render_chat_messages(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
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
            "system": "yellow",
            "user": "cyan",
            "assistant": "green",
            "tool": "magenta",
            "function": "magenta",
        }
        color = role_colors.get(role, "white")
        header = f"[bold {color}]{escape(role)}[/bold {color}]"

        # Show tool call info if present
        extra = ""
        if "tool_calls" in msg:
            calls = msg["tool_calls"]
            if isinstance(calls, list):
                names = [c.get("function", {}).get("name", "?") if isinstance(c, dict) else "?" for c in calls]
                extra = f"\n[dim]Tool calls: {', '.join(names)}[/dim]"

        text = f"{header}\n{escape(str(content))}{extra}"
        widgets.append(Static(text, classes=f"chat-message chat-{role}"))

    return Vertical(*widgets, classes="field-chat")


def _render_step_list(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
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


def _render_flat_dict(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    table = DataTable(classes="field-flat-dict")
    table.add_columns("Field", "Value")
    for k, v in value.items():
        table.add_row(str(k), str(v))
    return table


def _render_nested_dict(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
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


def _render_simple_list(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    items = [f"  [dim]{BULLET}[/dim] {escape(str(item))}" for item in value]
    return Static("\n".join(items), classes="field-list")


BULLET = "\u2022"


def _render_table(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
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


def _render_json_raw(value: Any, field_name: str, sub_fields: dict[str, str]) -> Widget:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    syntax = Syntax(text, "json", theme="monokai", line_numbers=False)
    return Static(syntax, classes="field-json")


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
    ValueType.SIMPLE_LIST: _render_simple_list,
    ValueType.TABLE: _render_table,
    ValueType.JSON_RAW: _render_json_raw,
}
