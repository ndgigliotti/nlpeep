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

_AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"


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


def _build_sp_token_provider(tenant_id: str, client_id: str, client_secret: str):
    """Build an azure-identity token provider from service principal credentials.

    Requires ``azure-identity``: uv pip install azure-identity
    """
    try:
        from azure.identity import ClientSecretCredential, get_bearer_token_provider
    except ImportError as exc:
        raise ImportError(
            "azure-identity is required for Azure AD auth. "
            "Install it with: uv pip install azure-identity"
        ) from exc

    cred = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )
    return get_bearer_token_provider(cred, _AZURE_SCOPE)


def speak(text: str) -> None:
    """Generate speech from text and play it, blocking until complete."""
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

    # Azure AD service principal: build token provider from tenant/client/secret
    tenant_id = config.pop("azure_tenant_id", None)
    client_id = config.pop("azure_client_id", None)
    client_secret = config.pop("azure_client_secret", None)
    if tenant_id and client_id and client_secret:
        kwargs["azure_ad_token_provider"] = _build_sp_token_provider(
            tenant_id, client_id, client_secret
        )

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
