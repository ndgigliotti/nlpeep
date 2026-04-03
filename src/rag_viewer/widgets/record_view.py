from __future__ import annotations

import json

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, TabbedContent, TabPane

from rag_viewer.data import Record
from rag_viewer.schema import FieldRole, SchemaMapping
from rag_viewer.widgets.field_panel import FieldPanel


class RecordContent(Widget):
    """Composable widget that builds tabbed content for a single record.

    We need this because TabbedContent requires TabPanes to be yielded
    during compose -- they can't be passed to the constructor.
    """

    DEFAULT_CSS = """
    RecordContent {
        height: 1fr;
    }
    """

    def __init__(
        self,
        record: Record,
        mapping: SchemaMapping,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.record = record
        self.mapping = mapping

    def compose(self) -> ComposeResult:
        record = self.record
        mapping = self.mapping
        active_roles = mapping.active_roles()
        tab_roles = [r for r in active_roles if r not in (FieldRole.QUERY, FieldRole.ID)]

        has_response = FieldRole.RESPONSE in active_roles
        has_ground_truth = FieldRole.GROUND_TRUTH in active_roles
        do_comparison = has_response and has_ground_truth

        with TabbedContent():
            for role in tab_roles:
                if do_comparison and role in (FieldRole.RESPONSE, FieldRole.GROUND_TRUTH):
                    if role == FieldRole.GROUND_TRUTH:
                        continue
                    yield self._build_comparison_pane(record, mapping)
                    continue

                role_mapping = mapping.get_mapping(role)
                if not role_mapping:
                    continue

                value = mapping.resolve(record.data, role)
                if value is None:
                    continue

                yield TabPane(
                    role_mapping.display_name,
                    FieldPanel(
                        field_name=role_mapping.json_path,
                        value=value,
                        role=role,
                        sub_fields=role_mapping.sub_fields,
                        show_label=False,
                    ),
                )

            # Unmapped fields tab
            unmapped = mapping.unmapped_fields(record.data)
            if unmapped:
                panels = [
                    FieldPanel(field_name=k, value=v, role=FieldRole.UNMAPPED)
                    for k, v in unmapped.items()
                ]
                yield TabPane("Details", Vertical(*panels))

            # Raw JSON tab
            raw = json.dumps(record.data, indent=2, ensure_ascii=False, default=str)
            raw_syntax = Syntax(raw, "json", theme="monokai", line_numbers=True)
            yield TabPane("Raw", Static(raw_syntax))

    def _build_comparison_pane(self, record: Record, mapping: SchemaMapping) -> TabPane:
        response_val = mapping.resolve(record.data, FieldRole.RESPONSE)
        gt_val = mapping.resolve(record.data, FieldRole.GROUND_TRUTH)

        response_mapping = mapping.get_mapping(FieldRole.RESPONSE)
        gt_mapping = mapping.get_mapping(FieldRole.GROUND_TRUTH)

        left = Vertical(
            Static("[bold]Response[/bold]"),
            FieldPanel(
                field_name=response_mapping.json_path if response_mapping else "response",
                value=response_val or "",
                role=FieldRole.RESPONSE,
                show_label=False,
            ),
            classes="comparison-pane",
        )
        right = Vertical(
            Static("[bold]Ground Truth[/bold]"),
            FieldPanel(
                field_name=gt_mapping.json_path if gt_mapping else "ground_truth",
                value=gt_val or "",
                role=FieldRole.GROUND_TRUTH,
                show_label=False,
            ),
            classes="comparison-pane",
        )

        return TabPane("Response vs Ground Truth", Horizontal(left, right))


class RecordView(Widget):
    """Main content area: shows one record with dynamic tabs based on schema mapping."""

    DEFAULT_CSS = """
    RecordView {
        width: 1fr;
    }
    RecordView #query-header {
        background: $primary-background;
        padding: 1 2;
        margin: 0 0 0 0;
        border: solid $primary-background-lighten-2;
        max-height: 6;
    }
    RecordView #record-content {
        height: 1fr;
    }
    RecordView .comparison-pane {
        width: 1fr;
        border-right: solid $primary-background-lighten-1;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._record: Record | None = None
        self._mapping: SchemaMapping = SchemaMapping()

    def compose(self) -> ComposeResult:
        yield Static("[dim]Load a file to begin[/dim]", id="query-header")
        yield VerticalScroll(id="record-content")

    def load_record(self, record: Record, mapping: SchemaMapping) -> None:
        self._record = record
        self._mapping = mapping

        # Update query header
        header = self.query_one("#query-header", Static)
        query_val = mapping.resolve(record.data, FieldRole.QUERY)
        if query_val:
            header.update(f"[bold]Q:[/bold] {query_val}")
        else:
            header.update(f"[dim]Record {record.index}[/dim]")

        # Rebuild tabbed content
        content = self.query_one("#record-content", VerticalScroll)
        content.remove_children()
        content.mount(RecordContent(record, mapping))
