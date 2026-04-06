from __future__ import annotations

import json
from typing import Any

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, TabbedContent, TabPane

from nlpeep.data import Record
from nlpeep.schema import FieldArchetype, FieldRole, SchemaMapping
from nlpeep.widgets.field_panel import FieldPanel


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
        tab_roles = [
            r for r in active_roles if r not in (FieldRole.QUERY, FieldRole.INPUT, FieldRole.ID)
        ]

        # Detect aligned pairs (e.g. NER tokens + tags)
        aligned = mapping.aligned_pairs()
        paired_paths = {p for am, bm in aligned for p in (am.json_path, bm.json_path)}

        has_response = FieldRole.RESPONSE in active_roles
        has_ground_truth = FieldRole.GROUND_TRUTH in active_roles
        do_comparison = has_response and has_ground_truth
        # Skip comparison when ground-truth is part of an aligned pair
        if do_comparison:
            gt_mappings = mapping.get_all_for_role(FieldRole.GROUND_TRUTH)
            if all(m.json_path in paired_paths for m in gt_mappings):
                do_comparison = False

        with TabbedContent():
            # Aligned pairs get a dedicated tab
            for a_m, b_m in aligned:
                a_val = record.get_path(a_m.json_path)
                b_val = record.get_path(b_m.json_path)
                if a_val is not None and b_val is not None:
                    tab_label = f"{a_m.json_path} / {b_m.json_path}"
                    yield TabPane(
                        tab_label,
                        FieldPanel(
                            field_name=tab_label,
                            value={a_m.json_path: a_val, b_m.json_path: b_val},
                            role=FieldRole.UNMAPPED,
                            archetype=FieldArchetype.ALIGNED_PAIR,
                            show_label=False,
                        ),
                    )

            # Primary input text gets its own tab (header is just a preview).
            for primary_role in (FieldRole.QUERY, FieldRole.INPUT):
                primary_mapping = mapping.get_mapping(primary_role)
                if not primary_mapping:
                    continue
                if primary_mapping.json_path in paired_paths:
                    break  # shown in aligned pair tab
                val = mapping.resolve(record.data, primary_role)
                if val is not None:
                    yield TabPane(
                        primary_role.display_name,
                        FieldPanel(
                            field_name=primary_mapping.json_path,
                            value=val,
                            role=primary_role,
                            show_label=False,
                        ),
                    )
                break  # only one primary role

            for role in tab_roles:
                # Skip roles whose fields are entirely consumed by tagged sequences
                all_for_role = mapping.get_all_for_role(role)
                if all_for_role and all(m.json_path in paired_paths for m in all_for_role):
                    continue

                if do_comparison and role in (FieldRole.RESPONSE, FieldRole.GROUND_TRUTH):
                    if role == FieldRole.GROUND_TRUTH:
                        continue
                    yield self._build_comparison_pane(record, mapping)
                    continue

                if not all_for_role:
                    continue

                if len(all_for_role) > 1:
                    # Multiple fields mapped to this role -- group into one dict
                    grouped_values: dict[str, Any] = {}
                    grouped_sub_fields: dict[str, str] = {}
                    for m in all_for_role:
                        val = record.get_path(m.json_path)
                        if val is not None:
                            grouped_values[m.json_path] = val
                            if m.metric_scale is not None:
                                grouped_sub_fields[m.json_path] = m.metric_scale.value
                    if grouped_values:
                        yield TabPane(
                            role.display_name,
                            FieldPanel(
                                field_name=role.display_name,
                                value=grouped_values,
                                role=role,
                                sub_fields=grouped_sub_fields,
                                show_label=False,
                            ),
                        )
                    continue

                role_mapping = all_for_role[0]
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
                        metric_scale=role_mapping.metric_scale,
                    ),
                )

            # Unmapped fields tab (exclude tagged-sequence paired fields)
            unmapped = mapping.unmapped_fields(record.data)
            for p in paired_paths:
                unmapped.pop(p, None)
            if unmapped:
                panels = [
                    FieldPanel(
                        field_name=k,
                        value=v,
                        role=FieldRole.UNMAPPED,
                        archetype=mapping.archetype_for_path(k),
                    )
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
        input_val = mapping.resolve(record.data, FieldRole.INPUT) if not query_val else None
        primary_val = query_val or input_val
        if primary_val:
            prefix = "Q:" if query_val else "Input:"
            display_val = primary_val
            if isinstance(display_val, str) and len(display_val) > 500:
                display_val = display_val[:500] + "..."
            header.update(f"[bold]{prefix}[/bold] {display_val}")
        else:
            header.update(f"[dim]Record {record.index}[/dim]")

        # Rebuild tabbed content
        content = self.query_one("#record-content", VerticalScroll)
        content.remove_children()
        content.mount(RecordContent(record, mapping))
