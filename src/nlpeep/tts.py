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

# Scope required for Azure Cognitive Services AD tokens
_AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"

# Maps config shorthand -> azure.identity credential class names
_AZURE_CREDENTIAL_NAMES: dict[str, str] = {
    "default": "DefaultAzureCredential",
    "cli": "AzureCliCredential",
    "env": "EnvironmentCredential",
    "workload": "WorkloadIdentityCredential",
    "managed": "ManagedIdentityCredential",
}


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


def _build_azure_token_provider(value: str):
    """Build an azure-identity token provider callable from a credential type string.

    ``value`` should be one of: default, cli, env, workload, managed.
    Requires ``azure-identity``: uv pip install azure-identity
    """
    try:
        import azure.identity as az_id
    except ImportError as exc:
        raise ImportError(
            "azure-identity is required for Azure AD token providers. "
            "Install it with: uv pip install azure-identity"
        ) from exc

    cred_name = _AZURE_CREDENTIAL_NAMES.get(value.lower(), "DefaultAzureCredential")
    cred_cls = getattr(az_id, cred_name)
    return az_id.get_bearer_token_provider(cred_cls(), _AZURE_SCOPE)


def speak(text: str) -> None:
    """Generate speech from text and play it non-blocking."""
    import litellm
    from playsound3 import playsound

    global _audio_file

    stop()

    config = _get_tts_config()

    kwargs: dict[str, Any] = {
        "model": config.pop("model", "openai/tts-1"),
        "voice": config.pop("voice", "alloy"),
        "input": text,
    }

    # Float-coerce numeric fields
    for key in ("speed", "timeout"):
        val = config.pop(key, None)
        if val is not None:
            kwargs[key] = float(val)

    # Azure AD token provider: convert string shorthand -> callable
    provider_val = config.pop("azure_ad_token_provider", None)
    if provider_val:
        kwargs["azure_ad_token_provider"] = _build_azure_token_provider(str(provider_val))

    # Pass through remaining keys (api_key, api_base, api_version, response_format, etc.)
    for k, v in config.items():
        if v != "" and v is not None:
            kwargs[k] = v

    response_format = kwargs.get("response_format", "mp3")
    response = litellm.speech(**kwargs)

    fd, tmp_path = tempfile.mkstemp(suffix=f".{response_format}")
    os.close(fd)
    _audio_file = Path(tmp_path)
    response.stream_to_file(_audio_file)

    try:
        # block=True so the caller's thread waits; stopsound() can interrupt it.
        # The finally clears state when playback ends naturally or is interrupted.
        playsound(str(_audio_file), block=True)
    finally:
        if _audio_file is not None:
            with contextlib.suppress(OSError):
                _audio_file.unlink(missing_ok=True)
        _audio_file = None


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
