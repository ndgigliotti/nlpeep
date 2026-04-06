from __future__ import annotations

import contextlib

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from nlpeep.user_config import load_user_config, save_user_config, user_config_path

_VOICE_OPTIONS: list[tuple[str, str]] = [
    ("alloy", "alloy"),
    ("ash", "ash"),
    ("coral", "coral"),
    ("echo", "echo"),
    ("fable", "fable"),
    ("nova", "nova"),
    ("onyx", "onyx"),
    ("sage", "sage"),
    ("shimmer", "shimmer"),
    ("custom...", "__custom__"),
]

_FORMAT_OPTIONS: list[tuple[str, str]] = [
    ("mp3 (default)", "mp3"),
    ("opus", "opus"),
    ("aac", "aac"),
    ("flac", "flac"),
    ("wav", "wav"),
    ("pcm", "pcm"),
]


class TTSModal(ModalScreen[bool]):
    """Modal for configuring TTS provider settings."""

    DEFAULT_CSS = """
    TTSModal {
        align: center middle;
    }
    TTSModal #modal-container {
        width: 76;
        max-height: 90%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    TTSModal #modal-title {
        text-align: center;
        text-style: bold;
        padding: 0 0 1 0;
    }
    TTSModal .section-header {
        text-style: bold dim;
        padding: 1 0 0 0;
        border-top: solid $primary-background-lighten-1;
        margin-top: 1;
    }
    TTSModal .field-label {
        padding: 1 0 0 0;
        color: $text-muted;
        height: 2;
    }
    TTSModal .hint {
        padding: 0 0 0 1;
        color: $text-muted;
        text-style: italic;
    }
    TTSModal #modal-buttons {
        height: 3;
        align: center middle;
        padding: 1 0 0 0;
    }
    TTSModal #modal-buttons Button {
        margin: 0 1;
    }
    TTSModal #custom-voice-row {
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        config = load_user_config().get("tts", {})

        current_voice = config.get("voice", "alloy")
        known_voices = {v for _, v in _VOICE_OPTIONS if v != "__custom__"}
        voice_select_val = "__custom__" if current_voice not in known_voices else current_voice

        current_format = config.get("response_format", "mp3")

        with Vertical(id="modal-container"):
            yield Static("TTS Configuration", id="modal-title")
            with VerticalScroll():
                yield Label("Model (litellm format):", classes="field-label")
                yield Input(
                    value=config.get("model", "openai/tts-1"),
                    placeholder="openai/tts-1",
                    id="input-model",
                )
                yield Static(
                    "e.g.  openai/tts-1  |  openai/tts-1-hd  |  azure/<deployment>",
                    classes="hint",
                )

                yield Label("Voice:", classes="field-label")
                yield Select(
                    _VOICE_OPTIONS,
                    value=voice_select_val,
                    allow_blank=False,
                    id="select-voice",
                )
                with Horizontal(id="custom-voice-row"):
                    yield Input(
                        value=current_voice if voice_select_val == "__custom__" else "",
                        placeholder="enter voice name",
                        id="input-custom-voice",
                    )
                self._update_custom_row_visibility(voice_select_val)

                yield Label("Speed (0.25 - 4.0, blank for default):", classes="field-label")
                speed = config.get("speed")
                yield Input(
                    value=str(speed) if speed is not None else "",
                    placeholder="1.0",
                    id="input-speed",
                )

                yield Label("Response Format:", classes="field-label")
                yield Select(
                    _FORMAT_OPTIONS,
                    value=current_format,
                    allow_blank=False,
                    id="select-format",
                )

                yield Static("API", classes="section-header")

                yield Label("API Key:", classes="field-label")
                yield Input(
                    value=config.get("api_key", ""),
                    placeholder="os.environ/OPENAI_API_KEY",
                    id="input-api-key",
                )
                yield Static(
                    "Use  os.environ/VAR_NAME  to read from environment",
                    classes="hint",
                )

                yield Label("API Base URL:", classes="field-label")
                yield Input(
                    value=config.get("api_base", ""),
                    placeholder="https://api.openai.com/v1",
                    id="input-api-base",
                )

                yield Label("API Version (Azure):", classes="field-label")
                yield Input(
                    value=config.get("api_version", ""),
                    placeholder="2024-05-01-preview",
                    id="input-api-version",
                )

                yield Label("Timeout (seconds, blank for default):", classes="field-label")
                timeout = config.get("timeout")
                yield Input(
                    value=str(timeout) if timeout is not None else "",
                    placeholder="30",
                    id="input-timeout",
                )

                yield Static(
                    f"For Azure AD service principal auth (azure_tenant_id, azure_client_id,\n"
                    f"azure_client_secret), edit {user_config_path()} directly.",
                    classes="hint",
                )

            with Horizontal(id="modal-buttons"):
                yield Button("Save", variant="primary", id="btn-save")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def _update_custom_row_visibility(self, voice_val: str) -> None:
        try:
            row = self.query_one("#custom-voice-row")
            row.display = voice_val == "__custom__"
        except Exception:
            pass

    @on(Select.Changed, "#select-voice")
    def _on_voice_changed(self, event: Select.Changed) -> None:
        self._update_custom_row_visibility(str(event.value))

    @on(Button.Pressed, "#btn-save")
    def _on_save(self) -> None:
        config = load_user_config()
        tts: dict = {}

        model = self.query_one("#input-model", Input).value.strip()
        if model:
            tts["model"] = model

        voice_select = self.query_one("#select-voice", Select)
        if voice_select.value == "__custom__":
            custom = self.query_one("#input-custom-voice", Input).value.strip()
            if custom:
                tts["voice"] = custom
        elif voice_select.value and voice_select.value != Select.BLANK:
            tts["voice"] = str(voice_select.value)

        speed_str = self.query_one("#input-speed", Input).value.strip()
        if speed_str:
            with contextlib.suppress(ValueError):
                tts["speed"] = float(speed_str)

        fmt = self.query_one("#select-format", Select)
        if fmt.value and fmt.value != Select.BLANK and fmt.value != "mp3":
            tts["response_format"] = str(fmt.value)

        api_key = self.query_one("#input-api-key", Input).value.strip()
        if api_key:
            tts["api_key"] = api_key

        api_base = self.query_one("#input-api-base", Input).value.strip()
        if api_base:
            tts["api_base"] = api_base

        api_version = self.query_one("#input-api-version", Input).value.strip()
        if api_version:
            tts["api_version"] = api_version

        timeout_str = self.query_one("#input-timeout", Input).value.strip()
        if timeout_str:
            with contextlib.suppress(ValueError):
                tts["timeout"] = float(timeout_str)

        # Preserve any advanced keys the user set directly in the config file
        # (e.g. azure_tenant_id, azure_client_id, azure_client_secret)
        existing_tts = load_user_config().get("tts", {})
        for k, v in existing_tts.items():
            if k not in tts:
                tts[k] = v

        config["tts"] = tts
        path = save_user_config(config)
        self.app.notify(f"TTS config saved to {path}")
        self.dismiss(True)

    @on(Button.Pressed, "#btn-cancel")
    def _on_cancel(self) -> None:
        self.dismiss(False)
