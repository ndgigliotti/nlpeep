"""Optional TTS (text-to-speech) support for nlpeep.

Requires the 'tts' extra: uv pip install nlpeep[tts]
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_audio_file: Path | None = None
_playback_proc: subprocess.Popen[bytes] | None = None


def is_available() -> bool:
    """Check whether TTS dependencies (litellm, playsound3) are installed."""
    try:
        import litellm  # noqa: F401
    except ImportError:
        return False
    if not _is_wsl():
        try:
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


def _is_wsl() -> bool:
    """Detect WSL by checking the kernel release string."""
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def _play_audio(path: Path) -> None:
    """Play an audio file, handling WSL2 and native Linux/macOS."""
    global _playback_proc

    # WSL2: use Windows-side PowerShell to play audio
    if _is_wsl() and shutil.which("powershell.exe"):
        win_path = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
        _playback_proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                (f"$p = New-Object System.Media.SoundPlayer '{win_path}';$p.PlaySync()"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    # Native: try playsound3
    try:
        from playsound3 import playsound

        playsound(str(path), block=False)
        return
    except Exception:
        pass

    # Fallback: try system players
    for cmd in ("ffplay", "mpv", "afplay", "aplay"):
        if shutil.which(cmd):
            args = [cmd]
            if cmd == "ffplay":
                args.extend(["-nodisp", "-autoexit", "-loglevel", "quiet"])
            elif cmd == "mpv":
                args.extend(["--no-video", "--really-quiet"])
            args.append(str(path))
            _playback_proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return

    raise RuntimeError("No audio backend found. Install ffplay, mpv, or fix playsound3.")


def speak(text: str) -> None:
    """Generate speech from text and play it non-blocking."""
    import litellm

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

    _play_audio(_audio_file)


def stop() -> None:
    """Stop current playback and clean up temp file."""
    global _audio_file, _playback_proc

    if _playback_proc is not None:
        with contextlib.suppress(OSError):
            _playback_proc.terminate()
        _playback_proc = None

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
    if _playback_proc is not None and _playback_proc.poll() is None:
        return True
    return _audio_file is not None
