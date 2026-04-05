from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, ListView, ListItem, Static

from nlpeep.data import RecordStore
from nlpeep.schema import SchemaMapping


class RecordNavigator(Widget):
    """Left panel: searchable list of records."""

    DEFAULT_CSS = """
    RecordNavigator {
        width: 36;
        border-right: solid $primary-background-lighten-2;
    }
    RecordNavigator #nav-search {
        margin: 0 0 0 0;
        border: none;
        background: $surface;
    }
    RecordNavigator #nav-list {
        height: 1fr;
    }
    RecordNavigator .nav-item-label {
        padding: 0 1;
    }
    RecordNavigator .nav-item-index {
        color: $text-muted;
        padding: 0 1;
    }
    """

    class RecordSelected(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(
        self,
        store: RecordStore,
        mapping: SchemaMapping,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.store = store
        self.mapping = mapping
        self._filtered_indices: list[int] = list(range(len(store)))

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search records...", id="nav-search")
        yield ListView(
            *self._build_items(self._filtered_indices),
            id="nav-list",
        )

    def _build_items(self, indices: list[int]) -> list[ListItem]:
        label_path = self.mapping.label_path()
        items = []
        for idx in indices:
            record = self.store[idx]
            label = record.label(label_path)
            item = ListItem(
                Static(
                    f"[dim $accent]{idx:>3}[/dim $accent]  [$text]{label}[/$text]",
                    classes="nav-item-label",
                ),
            )
            item._record_index = idx  # type: ignore[attr-defined]
            items.append(item)
        return items

    @on(Input.Changed, "#nav-search")
    def _on_search(self, event: Input.Changed) -> None:
        query = event.value.strip()
        if query:
            self._filtered_indices = self.store.search(query)
        else:
            self._filtered_indices = list(range(len(self.store)))
        self._rebuild_list()

    @on(Input.Submitted, "#nav-search")
    def _on_search_submit(self, event: Input.Submitted) -> None:
        """After pressing Enter in search, focus the list for j/k navigation."""
        self.query_one("#nav-list", ListView).focus()

    @on(ListView.Selected, "#nav-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        item = event.item
        idx = getattr(item, "_record_index", 0)
        self.post_message(self.RecordSelected(idx))

    def _rebuild_list(self) -> None:
        list_view = self.query_one("#nav-list", ListView)
        list_view.clear()
        for item in self._build_items(self._filtered_indices):
            list_view.append(item)

    def refresh_labels(self, mapping: SchemaMapping) -> None:
        self.mapping = mapping
        self._rebuild_list()

    def select_index(self, idx: int) -> None:
        """Programmatically highlight a record in the list."""
        list_view = self.query_one("#nav-list", ListView)
        try:
            position = self._filtered_indices.index(idx)
            list_view.index = position
        except ValueError:
            pass
