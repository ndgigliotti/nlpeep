"""Optional TTS (text-to-speech) support for nlpeep.

Requires the 'tts' extra: uv pip install nlpeep[tts]
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

_audio_file: Path | None = None


def is_available() -> bool:
    """Check whether TTS dependencies (litellm, playsound3) are installed."""
    try:
        import litellm  # noqa: F401
        import playsound3  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_env(value: Any) -> Any:
    """Resolve ``os.environ/VAR_NAME`` strings to actual env var values."""
    if isinstance(value, str) and value.startswith("os.environ/"):
        var_name = value[len("os.environ/") :]
        return os.environ.get(var_name, "")
    return value


def _get_tts_config() -> dict[str, Any]:
    """Read [tts] section from user config with env var resolution."""
    from nlpeep.user_config import load_user_config

    config = load_user_config()
    tts = config.get("tts", {})
    return {k: _resolve_env(v) for k, v in tts.items()}


def speak(text: str) -> None:
    """Generate speech from text and play it non-blocking."""
    import litellm
    from playsound3 import playsound

    global _audio_file

    stop()

    config = _get_tts_config()

    kwargs: dict[str, Any] = {
        "model": config.get("model", "openai/tts-1"),
        "voice": config.get("voice", "alloy"),
        "input": text,
    }

    if config.get("api_key"):
        kwargs["api_key"] = config["api_key"]
    if config.get("api_base"):
        kwargs["api_base"] = config["api_base"]

    speed = config.get("speed")
    if speed is not None:
        kwargs["speed"] = float(speed)

    response = litellm.speech(**kwargs)

    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    _audio_file = Path(tmp_path)
    response.stream_to_file(_audio_file)

    playsound(str(_audio_file), block=False)


def stop() -> None:
    """Stop current playback and clean up temp file."""
    global _audio_file

    try:
        from playsound3 import stopsound

        stopsound()
    except (ImportError, Exception):
        pass

    if _audio_file is not None:
        with contextlib.suppress(OSError):
            _audio_file.unlink(missing_ok=True)
    _audio_file = None


def is_speaking() -> bool:
    """Check if audio playback was started and not yet stopped."""
    return _audio_file is not None
