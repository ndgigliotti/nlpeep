from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from rag_viewer.schema import SchemaMapping


def find_config(data_path: Path, explicit: Path | None = None) -> Path | None:
    """Discover the config file to use, in priority order."""
    if explicit and explicit.exists():
        return explicit

    parent = data_path.parent
    stem = data_path.stem

    # Per-file config
    per_file = parent / f".rag-viewer.{stem}.toml"
    if per_file.exists():
        return per_file

    # Directory config
    dir_config = parent / ".rag-viewer.toml"
    if dir_config.exists():
        return dir_config

    # CWD config
    cwd_config = Path.cwd() / ".rag-viewer.toml"
    if cwd_config.exists() and cwd_config != dir_config:
        return cwd_config

    return None


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_mapping(data_path: Path, explicit_config: Path | None = None) -> SchemaMapping | None:
    """Try to load a SchemaMapping from config. Returns None if no config found."""
    config_path = find_config(data_path, explicit_config)
    if not config_path:
        return None
    config = load_config(config_path)
    return SchemaMapping.from_config(config)


def save_config(data_path: Path, mapping: SchemaMapping) -> Path:
    """Save mapping as a per-file config. Returns the path written."""
    parent = data_path.parent
    stem = data_path.stem
    config_path = parent / f".rag-viewer.{stem}.toml"
    config = mapping.to_config()

    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)

    return config_path
