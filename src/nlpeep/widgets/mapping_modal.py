from __future__ import annotations

import contextlib
from collections import Counter

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, Static

from nlpeep.config import save_config
from nlpeep.data import RecordStore
from nlpeep.schema import FieldMapping, FieldRole, MetricScale, SchemaMapping

_ROLE_OPTIONS = [(role.display_name, role.value) for role in FieldRole]

_SCALE_OPTIONS: list[tuple[str, str]] = [
    ("Auto", "auto"),
    ("Unit [0,1]", MetricScale.UNIT.value),
    ("Percent [0,100]", MetricScale.PERCENT.value),
    ("Unknown", MetricScale.UNKNOWN.value),
]


class MappingModal(ModalScreen[SchemaMapping | None]):
    """Modal for configuring field-to-role mappings."""

    DEFAULT_CSS = """
    MappingModal {
        align: center middle;
    }
    MappingModal #modal-container {
        width: 110;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    MappingModal #modal-title {
        text-align: center;
        text-style: bold;
        padding: 0 0 1 0;
    }
    MappingModal .header-row {
        height: 1;
        padding: 0 1;
    }
    MappingModal .header-label {
        text-style: bold dim;
    }
    MappingModal .header-field {
        width: 36;
    }
    MappingModal .header-type {
        width: 14;
    }
    MappingModal .header-role {
        width: 30;
    }
    MappingModal .field-row {
        height: 3;
        padding: 0 1;
    }
    MappingModal .field-name {
        width: 36;
        padding: 1 1 0 0;
    }
    MappingModal .field-type {
        width: 14;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    MappingModal .role-select {
        width: 30;
    }
    MappingModal .scale-select {
        width: 20;
    }
    MappingModal .header-scale {
        width: 20;
    }
    MappingModal .confidence-tag {
        width: 14;
        padding: 1 0 0 1;
        color: $text-muted;
    }
    MappingModal .group-hint {
        width: 16;
        padding: 1 0 0 0;
        color: $text-muted;
    }
    MappingModal #modal-buttons {
        height: 3;
        align: center middle;
        padding: 1 0 0 0;
    }
    MappingModal #modal-buttons Button {
        margin: 0 1;
    }
    """

    class MappingSaved(Message):
        def __init__(self, mapping: SchemaMapping) -> None:
            super().__init__()
            self.mapping = mapping

    def __init__(
        self,
        store: RecordStore,
        mapping: SchemaMapping,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.store = store
        self.current_mapping = mapping
        self._selects: dict[str, Select[str]] = {}
        self._scale_selects: dict[str, Select[str]] = {}
        self._group_labels: dict[str, Label] = {}
        # Build confidence lookup from existing FieldMapping objects
        self._confidence: dict[str, float] = {}
        self._current_scales: dict[str, str] = {}
        for m in self.current_mapping.mappings:
            self._confidence[m.json_path] = m.confidence
            if m.metric_scale is not None:
                self._current_scales[m.json_path] = m.metric_scale.value

    def compose(self) -> ComposeResult:
        field_summary = self.store.field_summary()

        # Build a lookup of current assignments
        path_to_role: dict[str, str] = {}
        for m in self.current_mapping.mappings:
            path_to_role[m.json_path] = m.role.value

        with Vertical(id="modal-container"):
            yield Static("Field Mapping Configuration", id="modal-title")
            with VerticalScroll():
                # Header row with column labels
                with Horizontal(classes="header-row"):
                    yield Label(
                        "Field",
                        classes="header-label header-field",
                    )
                    yield Label(
                        "Type",
                        classes="header-label header-type",
                    )
                    yield Label(
                        "Role",
                        classes="header-label header-role",
                    )
                    yield Label(
                        "Scale",
                        classes="header-label header-scale",
                    )
                # Sort fields so shallower paths appear first, and
                # nested paths are visually grouped under their parent.
                sorted_fields = sorted(
                    field_summary.items(),
                    key=lambda item: (item[0].count("."), item[0]),
                )
                for field_name, types in sorted_fields:
                    type_str = ", ".join(sorted(types))
                    current_role = path_to_role.get(field_name, FieldRole.UNMAPPED.value)
                    # Use a sanitized ID: dots are not valid in CSS ids
                    safe_id = field_name.replace(".", "__")
                    select = Select(
                        _ROLE_OPTIONS,
                        value=current_role,
                        allow_blank=False,
                        classes="role-select",
                        id=f"select-{safe_id}",
                    )
                    self._selects[field_name] = select
                    # Confidence tag for auto-detected mappings
                    conf = self._confidence.get(field_name, 0.0)
                    conf_text = f"(auto {conf:.2f})" if conf > 0 and conf < 1.0 else ""
                    # Scale selector (visible only for metrics role)
                    scale_val = self._current_scales.get(field_name, "auto")
                    scale_select = Select(
                        _SCALE_OPTIONS,
                        value=scale_val,
                        allow_blank=False,
                        classes="scale-select",
                        id=f"scale-{safe_id}",
                    )
                    scale_select.display = current_role == FieldRole.METRICS.value
                    self._scale_selects[field_name] = scale_select
                    # Group hint label (updated reactively)
                    group_label = Label("", classes="group-hint", id=f"group-{safe_id}")
                    self._group_labels[field_name] = group_label
                    # Indent nested paths for visual grouping
                    depth = field_name.count(".")
                    display_name = ("  " * depth) + field_name
                    with Horizontal(classes="field-row"):
                        yield Label(display_name, classes="field-name")
                        yield Label(type_str, classes="field-type")
                        yield select
                        yield scale_select
                        yield Label(conf_text, classes="confidence-tag")
                        yield group_label

            with Horizontal(id="modal-buttons"):
                yield Button("Apply", variant="primary", id="btn-apply")
                yield Button("Save to File", variant="success", id="btn-save")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_mount(self) -> None:
        """Update group hints and scale visibility once all widgets are mounted."""
        self._update_group_hints()
        self._update_scale_visibility()

    @on(Select.Changed)
    def _on_role_changed(self, event: Select.Changed) -> None:
        """When any role Select changes, recalculate group hints and scale visibility."""
        self._update_group_hints()
        self._update_scale_visibility()

    def _update_group_hints(self) -> None:
        """Scan all selects, count roles, and update group-hint labels."""
        # Count how many fields share each role
        role_counts: Counter[str] = Counter()
        for select in self._selects.values():
            val = select.value
            if val != Select.BLANK and val != FieldRole.UNMAPPED.value:
                role_counts[str(val)] += 1

        # Update each group-hint label
        for field_name, select in self._selects.items():
            label = self._group_labels.get(field_name)
            if label is None:
                continue
            val = select.value
            if val == Select.BLANK or val == FieldRole.UNMAPPED.value:
                label.update("")
                continue
            total = role_counts[str(val)]
            if total > 1:
                others = total - 1
                label.update(f"(+{others} other{'s' if others > 1 else ''})")
            else:
                label.update("")

    def _update_scale_visibility(self) -> None:
        """Show scale selects only for fields with role == METRICS."""
        for field_name, role_select in self._selects.items():
            scale_select = self._scale_selects.get(field_name)
            if scale_select is None:
                continue
            scale_select.display = role_select.value == FieldRole.METRICS.value

    def _build_mapping(self) -> SchemaMapping:
        mappings: list[FieldMapping] = []
        for field_name, select in self._selects.items():
            role_val = select.value
            if role_val == Select.BLANK or role_val == FieldRole.UNMAPPED.value:
                continue
            role = FieldRole(role_val)
            # Carry over sub_fields from existing mapping if available
            existing = next(
                (m for m in self.current_mapping.mappings if m.json_path == field_name),
                None,
            )
            sub_fields = existing.sub_fields if existing else {}
            # Determine metric scale from the scale select
            metric_scale: MetricScale | None = None
            if role == FieldRole.METRICS:
                scale_select = self._scale_selects.get(field_name)
                if scale_select and scale_select.value != "auto":
                    with contextlib.suppress(ValueError):
                        metric_scale = MetricScale(scale_select.value)
                elif existing and existing.metric_scale is not None:
                    metric_scale = existing.metric_scale
            mappings.append(
                FieldMapping(
                    json_path=field_name,
                    role=role,
                    sub_fields=sub_fields,
                    confidence=1.0,
                    metric_scale=metric_scale,
                )
            )
        return SchemaMapping(mappings=mappings)

    @on(Button.Pressed, "#btn-apply")
    def _on_apply(self, event: Button.Pressed) -> None:
        mapping = self._build_mapping()
        self.dismiss(mapping)

    @on(Button.Pressed, "#btn-save")
    def _on_save(self, event: Button.Pressed) -> None:
        mapping = self._build_mapping()
        if self.store.path:
            path = save_config(self.store.path, mapping)
            self.app.notify(f"Saved to {path}")
        self.dismiss(mapping)

    @on(Button.Pressed, "#btn-cancel")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)
