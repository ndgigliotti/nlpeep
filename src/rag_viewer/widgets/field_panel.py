from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

from rag_viewer.renderers import classify_value, render_value
from rag_viewer.schema import FieldRole


class FieldPanel(Widget):
    """Renders a single field with a label and type-appropriate widget."""

    DEFAULT_CSS = """
    FieldPanel {
        height: auto;
        padding: 0 0 1 0;
    }
    FieldPanel .field-label {
        color: $text-muted;
        text-style: bold;
        padding: 0 0 0 0;
    }
    """

    def __init__(
        self,
        field_name: str,
        value: Any,
        role: FieldRole = FieldRole.UNMAPPED,
        sub_fields: dict[str, str] | None = None,
        show_label: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.field_name = field_name
        self.value = value
        self.role = role
        self.sub_fields = sub_fields or {}
        self.show_label = show_label

    def compose(self) -> ComposeResult:
        try:
            vtype = classify_value(self.value, self.field_name, self.role)
            widget = render_value(self.value, vtype, self.field_name, self.sub_fields)
        except Exception:
            import json
            fallback = json.dumps(self.value, indent=2, ensure_ascii=False, default=str)
            widget = Static(fallback)

        if self.show_label:
            yield Vertical(
                Static(f"[bold]{self.field_name}[/bold]", classes="field-label"),
                widget,
            )
        else:
            yield widget
