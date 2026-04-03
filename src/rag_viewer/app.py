from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Static

from rag_viewer.config import load_mapping
from rag_viewer.data import RecordStore
from rag_viewer.schema import SchemaMapping
from rag_viewer.widgets.mapping_modal import MappingModal
from rag_viewer.widgets.navigator import RecordNavigator
from rag_viewer.widgets.record_view import RecordView


class RagViewerApp(App):
    CSS_PATH = "styles/app.tcss"
    TITLE = "rag-viewer"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+m", "open_mapping", "Mapping"),
        Binding("ctrl+r", "reload", "Reload"),
        Binding("ctrl+f", "focus_search", "Search"),
        Binding("escape", "unfocus_search", "Unfocus", show=False),
        Binding("j", "next_record", "Next", show=False),
        Binding("k", "prev_record", "Prev", show=False),
    ]

    def __init__(
        self,
        path: Path,
        config_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._path = path
        self._config_path = config_path
        self._store: RecordStore | None = None
        self._mapping: SchemaMapping = SchemaMapping()
        self._current_index: int = 0
        self._auto_detected: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(id="main-container")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_data()

    def _load_data(self) -> None:
        self._store = RecordStore.load(self._path)

        if not self._store.records:
            self.notify("No valid records found", severity="error")
            return

        # Try to load mapping from config, fall back to auto-detect
        mapping = load_mapping(self._path, self._config_path)
        if mapping and mapping.mappings:
            self._mapping = mapping
            self._auto_detected = False
        else:
            self._mapping = SchemaMapping.auto_detect(self._store.sample())
            self._auto_detected = True

        # Build the UI
        container = self.query_one("#main-container", Horizontal)
        container.remove_children()

        nav = RecordNavigator(self._store, self._mapping)
        view = RecordView()
        container.mount(nav, view)

        # Defer first record load + focus until children are composed
        def _after_mount() -> None:
            self._select_record(0)
            self._update_status()
            try:
                from textual.widgets import ListView
                self.query_one("#nav-list", ListView).focus()
            except Exception:
                pass
            if self._auto_detected:
                roles = self._mapping.active_roles()
                role_names = ", ".join(r.display_name for r in roles)
                self.notify(
                    f"Auto-detected: {role_names}. Press Ctrl+M to adjust.",
                    timeout=5,
                )

        self.call_after_refresh(_after_mount)

    def _select_record(self, index: int) -> None:
        if not self._store or index < 0 or index >= len(self._store):
            return
        self._current_index = index
        record = self._store[index]
        try:
            view = self.query_one(RecordView)
            view.load_record(record, self._mapping)
        except Exception:
            pass
        self._update_status()

    def _update_status(self) -> None:
        if not self._store:
            return
        status = self.query_one("#status-bar", Static)
        parts = [
            f"{self._current_index + 1}/{len(self._store)} records",
            str(self._path.name),
        ]
        if self._store.skipped:
            parts.append(f"{self._store.skipped} skipped")
        if self._auto_detected:
            parts.append("auto-detected")
        status.update("  |  ".join(parts))

    @on(RecordNavigator.RecordSelected)
    def _on_record_selected(self, event: RecordNavigator.RecordSelected) -> None:
        self._select_record(event.index)

    def action_open_mapping(self) -> None:
        if not self._store:
            return

        def on_dismiss(result: SchemaMapping | None) -> None:
            if result is not None:
                self._mapping = result
                self._auto_detected = False
                # Refresh navigator labels
                try:
                    nav = self.query_one(RecordNavigator)
                    nav.refresh_labels(self._mapping)
                except Exception:
                    pass
                # Reload current record
                self._select_record(self._current_index)
                self._update_status()

        self.push_screen(
            MappingModal(self._store, self._mapping),
            on_dismiss,
        )

    def action_reload(self) -> None:
        self._load_data()
        self.notify("Reloaded")

    def action_focus_search(self) -> None:
        try:
            from textual.widgets import Input
            search = self.query_one("#nav-search", Input)
            search.focus()
        except Exception:
            pass

    def action_unfocus_search(self) -> None:
        """Move focus away from search input to the record list."""
        try:
            from textual.widgets import ListView
            list_view = self.query_one("#nav-list", ListView)
            list_view.focus()
        except Exception:
            pass

    def _input_focused(self) -> bool:
        from textual.widgets import Input
        return isinstance(self.focused, Input)

    def action_next_record(self) -> None:
        if self._input_focused():
            return
        if self._store:
            new_idx = min(self._current_index + 1, len(self._store) - 1)
            self._select_record(new_idx)
            try:
                nav = self.query_one(RecordNavigator)
                nav.select_index(new_idx)
            except Exception:
                pass

    def action_prev_record(self) -> None:
        if self._input_focused():
            return
        if self._store:
            new_idx = max(self._current_index - 1, 0)
            self._select_record(new_idx)
            try:
                nav = self.query_one(RecordNavigator)
                nav.select_index(new_idx)
            except Exception:
                pass
