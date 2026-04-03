from __future__ import annotations

from rich.markup import escape
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static


class ScoreBar(Widget):
    """A horizontal bar showing a 0-1 score with color coding."""

    DEFAULT_CSS = """
    ScoreBar {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        value: float,
        label: str = "",
        bar_width: int = 20,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.value = max(0.0, min(1.0, value))
        self.label = label
        self.bar_width = bar_width

    def compose(self) -> ComposeResult:
        filled = int(self.value * self.bar_width)
        empty = self.bar_width - filled

        if self.value >= 0.8:
            color = "#00ff88"
        elif self.value >= 0.5:
            color = "#ffb000"
        else:
            color = "#ff2d95"

        bar = f"[{color}]{'\u2588' * filled}[/{color}][#1e2a3a]{'\u2591' * empty}[/#1e2a3a]"
        label_part = f"[bold #ccddee]{escape(self.label)}[/bold #ccddee]  " if self.label else ""
        text = f"{label_part}{bar}  [bold {color}]{self.value:.3f}[/bold {color}]"
        yield Static(text)
