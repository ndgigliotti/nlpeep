from __future__ import annotations

import json
from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Collapsible


class DocCard(Widget):
    """Renders a single retrieved document with rank, score, and content."""

    DEFAULT_CSS = """
    DocCard {
        height: auto;
        margin: 0 0 1 0;
        padding: 0;
    }
    DocCard .doc-header {
        background: $surface;
        padding: 0 1;
    }
    DocCard .doc-body {
        padding: 0 2;
    }
    """

    MAX_PREVIEW_LINES = 12

    def __init__(
        self,
        doc: dict[str, Any],
        rank: int = 0,
        sub_fields: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.doc = doc
        self.rank = rank
        self.sub_fields = sub_fields or {}

    def compose(self) -> ComposeResult:
        text_key = self.sub_fields.get("text")
        score_key = self.sub_fields.get("score")
        source_key = self.sub_fields.get("source")

        # Auto-detect keys if not configured
        if not text_key:
            for k in ("text", "content", "passage", "chunk", "document", "page_content", "body"):
                if k in self.doc:
                    text_key = k
                    break
        if not score_key:
            for k in ("score", "relevance", "similarity", "distance", "relevance_score", "rank"):
                if k in self.doc:
                    score_key = k
                    break
        if not source_key:
            for k in ("source", "title", "url", "filename", "doc_id", "name", "id"):
                if k in self.doc:
                    source_key = k
                    break

        # Build header
        header_parts = [f"[bold]#{self.rank}[/bold]"]
        if score_key and score_key in self.doc:
            score = self.doc[score_key]
            if isinstance(score, (int, float)):
                if score >= 0.8:
                    color = "green"
                elif score >= 0.5:
                    color = "yellow"
                else:
                    color = "red"
                header_parts.append(f"[{color}]{score:.4f}[/{color}]")
            else:
                header_parts.append(str(score))
        if source_key and source_key in self.doc:
            header_parts.append(f"[dim]{escape(str(self.doc[source_key]))}[/dim]")

        yield Static("  ".join(header_parts), classes="doc-header")

        # Build body
        if text_key and text_key in self.doc:
            body_text = str(self.doc[text_key])
        else:
            # Show all fields not already shown in header
            shown = {text_key, score_key, source_key}
            remaining = {k: v for k, v in self.doc.items() if k not in shown}
            if remaining:
                body_text = json.dumps(remaining, indent=2, ensure_ascii=False, default=str)
            else:
                body_text = json.dumps(self.doc, indent=2, ensure_ascii=False, default=str)

        lines = body_text.split("\n")
        if len(lines) > self.MAX_PREVIEW_LINES:
            preview = "\n".join(lines[: self.MAX_PREVIEW_LINES])
            full = body_text
            yield Collapsible(
                Static(full, classes="doc-body"),
                title=f"{preview[:80]}... ({len(lines)} lines)",
                collapsed=True,
            )
        else:
            yield Static(body_text, classes="doc-body")
