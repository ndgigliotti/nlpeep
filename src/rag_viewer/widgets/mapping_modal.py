from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, Static

from rag_viewer.config import save_config
from rag_viewer.data import RecordStore
from rag_viewer.schema import FieldMapping, FieldRole, SchemaMapping


_ROLE_OPTIONS = [(role.display_name, role.value) for role in FieldRole]


class MappingModal(ModalScreen[SchemaMapping | None]):
    """Modal for configuring field-to-role mappings."""

    DEFAULT_CSS = """
    MappingModal {
        align: center middle;
    }
    MappingModal #modal-container {
        width: 80;
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
    MappingModal .field-row {
        height: 3;
        padding: 0 1;
    }
    MappingModal .field-name {
        width: 30;
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

    def compose(self) -> ComposeResult:
        field_summary = self.store.field_summary()

        # Build a lookup of current assignments
        path_to_role: dict[str, str] = {}
        for m in self.current_mapping.mappings:
            path_to_role[m.json_path] = m.role.value

        with Vertical(id="modal-container"):
            yield Static("Field Mapping Configuration", id="modal-title")
            with VerticalScroll():
                for field_name, types in field_summary.items():
                    type_str = ", ".join(sorted(types))
                    current_role = path_to_role.get(field_name, FieldRole.UNMAPPED.value)
                    select = Select(
                        _ROLE_OPTIONS,
                        value=current_role,
                        allow_blank=False,
                        classes="role-select",
                        id=f"select-{field_name}",
                    )
                    self._selects[field_name] = select
                    with Horizontal(classes="field-row"):
                        yield Label(field_name, classes="field-name")
                        yield Label(type_str, classes="field-type")
                        yield select

            with Horizontal(id="modal-buttons"):
                yield Button("Apply", variant="primary", id="btn-apply")
                yield Button("Save to File", variant="success", id="btn-save")
                yield Button("Cancel", variant="default", id="btn-cancel")

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
            mappings.append(FieldMapping(
                json_path=field_name,
                role=role,
                sub_fields=sub_fields,
                confidence=1.0,
            ))
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
