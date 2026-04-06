from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.theme import Theme
from textual.widgets import Footer, Header, Static

from nlpeep.config import load_mapping
from nlpeep.data import RecordStore
from nlpeep.schema import SchemaMapping
from nlpeep.widgets.mapping_modal import MappingModal
from nlpeep.widgets.navigator import RecordNavigator
from nlpeep.widgets.record_view import RecordView
from nlpeep.widgets.tts_modal import TTSModal


class NLPeepApp(App):
    CSS_PATH = "styles/app.tcss"
    TITLE = "NLPEEP"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("m", "open_mapping", "Mapping"),
        Binding("ctrl+r", "reload", "Reload"),
        Binding("ctrl+f", "focus_search", "Search"),
        Binding("escape", "unfocus_search", "Unfocus", show=False),
        Binding("ctrl+s", "speak", "Speak"),
        Binding("ctrl+t", "tts_settings", "TTS"),
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
        self.register_theme(
            Theme(
                name="neon-cyberpunk",
                primary="#00e5ff",
                secondary="#ff2d95",
                accent="#00e5ff",
                warning="#ffb000",
                error="#ff2d95",
                success="#00ff88",
                foreground="#e8f0ff",
                background="#0a0a0f",
                surface="#131820",
                panel="#0d1117",
                dark=True,
            )
        )
        from nlpeep.user_config import get_user_setting

        self.theme = get_user_setting("display", "theme", "neon-cyberpunk")
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
        self.sub_title = str(self._path.name)
        self._store = RecordStore.load(self._path)

        if not self._store.records:
            self.notify("No valid records found", severity="error")
            return

        # Attempt trace assembly for multi-span pipeline data
        self._store = self._store.maybe_assemble_traces()

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
                    f"Auto-detected: {role_names}. Press M to adjust.",
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
        if not self._store or self._input_focused():
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

    def action_speak(self) -> None:
        if self._input_focused():
            return
        from nlpeep import tts

        if not tts.is_available():
            self.notify("Install nlpeep[tts] for text-to-speech", severity="warning")
            return
        if tts.is_speaking():
            tts.stop()
            self.notify("Stopped")
            return
        text = self._get_speakable_text()
        if not text:
            self.notify("No text to speak", severity="warning")
            return
        self.notify("Speaking...")
        self._do_speak(text)

    def action_tts_settings(self) -> None:
        if self._input_focused():
            return
        self.push_screen(TTSModal())

    @work(thread=True, exclusive=True, group="tts")
    def _do_speak(self, text: str) -> None:
        from nlpeep import tts

        try:
            tts.speak(text)
        except Exception as exc:
            tts.stop()
            self.call_from_thread(self.notify, f"TTS error: {exc}", severity="error")

    def _get_speakable_text(self) -> str:
        """Extract text from the active tab's content."""
        parts: list[str] = []
        try:
            from textual.widgets import TabbedContent

            from nlpeep.widgets.doc_card import DocCard
            from nlpeep.widgets.field_panel import FieldPanel
            from nlpeep.widgets.record_view import RecordContent

            view = self.query_one(RecordView)
            content = view.query_one(RecordContent)
            tabbed = content.query_one(TabbedContent)
            active_pane = tabbed.active_pane
            if active_pane:
                for fp in active_pane.query(FieldPanel):
                    if isinstance(fp.value, str):
                        parts.append(fp.value)
                    elif isinstance(fp.value, dict):
                        str_vals = [str(v) for v in fp.value.values() if isinstance(v, str)]
                        if str_vals:
                            parts.append("\n".join(str_vals))
                for card in active_pane.query(DocCard):
                    text = card.speakable_text()
                    if text:
                        parts.append(text)
        except Exception:
            pass
        return "\n\n".join(parts)
