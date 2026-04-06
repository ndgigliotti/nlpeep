"""User-scoped configuration at ~/.config/nlpeep/config.toml (or platform equivalent)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from nlpeep.schema import SchemaMapping


def user_config_path() -> Path:
    """Platform-appropriate path to ~/.config/nlpeep/config.toml (or Windows equiv).

    On macOS, uses ~/.config/nlpeep/ (matching ruff/uv convention) instead of
    the platformdirs default of ~/Library/Application Support/.
    """
    if sys.platform == "darwin":
        return Path.home() / ".config" / "nlpeep" / "config.toml"

    import platformdirs

    return Path(platformdirs.user_config_dir("nlpeep")) / "config.toml"


def load_user_config() -> dict[str, Any]:
    """Load user config. Returns empty dict if file doesn't exist."""
    path = user_config_path()
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def save_user_config(config: dict[str, Any]) -> Path:
    """Write full config dict. Creates parent dirs. Returns path written."""
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(config, f)
    return path


def get_user_setting(section: str, key: str, default: Any = None) -> Any:
    """Read single value. e.g. get_user_setting("tts", "provider", "openai")."""
    config = load_user_config()
    return config.get(section, {}).get(key, default)


def get_default_mapping() -> SchemaMapping | None:
    """Build SchemaMapping from [defaults] section, or None if empty."""
    config = load_user_config()
    defaults = config.get("defaults", {})
    if not defaults or "mapping" not in defaults:
        return None
    result = SchemaMapping.from_config(defaults)
    return result if result.mappings else None
