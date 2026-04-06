"""Tests for user-scoped config module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nlpeep.user_config import (
    get_default_mapping,
    get_user_setting,
    load_user_config,
    save_user_config,
    user_config_path,
)

# -- user_config_path ---------------------------------------------------------


def test_path_linux(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    with patch("platformdirs.user_config_dir", return_value=str(tmp_path / ".config" / "nlpeep")):
        p = user_config_path()
    assert p == tmp_path / ".config" / "nlpeep" / "config.toml"


def test_path_macos(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("HOME", "/Users/testuser")
    p = user_config_path()
    assert p == Path("/Users/testuser/.config/nlpeep/config.toml")


def test_path_windows(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    appdata = str(tmp_path / "AppData" / "Roaming")
    monkeypatch.setenv("APPDATA", appdata)
    with patch(
        "platformdirs.user_config_dir",
        return_value=str(tmp_path / "AppData" / "Roaming" / "nlpeep"),
    ):
        p = user_config_path()
    assert p == tmp_path / "AppData" / "Roaming" / "nlpeep" / "config.toml"


# -- load / save round-trip ---------------------------------------------------


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nlpeep.user_config.user_config_path", lambda: tmp_path / "nope" / "config.toml"
    )
    assert load_user_config() == {}


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    cfg_path = tmp_path / "nlpeep" / "config.toml"
    monkeypatch.setattr("nlpeep.user_config.user_config_path", lambda: cfg_path)

    config = {
        "tts": {"provider": "openai", "voice": "alloy", "speed": 1.0},
        "display": {"theme": "neon-cyberpunk"},
    }
    returned_path = save_user_config(config)
    assert returned_path == cfg_path
    assert cfg_path.exists()

    loaded = load_user_config()
    assert loaded == config


# -- get_user_setting ----------------------------------------------------------


def test_get_user_setting_with_value(tmp_path, monkeypatch):
    cfg_path = tmp_path / "nlpeep" / "config.toml"
    monkeypatch.setattr("nlpeep.user_config.user_config_path", lambda: cfg_path)
    save_user_config({"tts": {"provider": "elevenlabs"}})

    assert get_user_setting("tts", "provider") == "elevenlabs"


def test_get_user_setting_default(tmp_path, monkeypatch):
    monkeypatch.setattr("nlpeep.user_config.user_config_path", lambda: tmp_path / "nope.toml")
    assert get_user_setting("tts", "provider", "openai") == "openai"


# -- get_default_mapping -------------------------------------------------------


def test_get_default_mapping_returns_schema_mapping(tmp_path, monkeypatch):
    cfg_path = tmp_path / "nlpeep" / "config.toml"
    monkeypatch.setattr("nlpeep.user_config.user_config_path", lambda: cfg_path)
    save_user_config(
        {
            "defaults": {
                "mapping": {"query": "question", "response": "answer"},
            },
        }
    )

    from nlpeep.schema import FieldRole

    mapping = get_default_mapping()
    assert mapping is not None
    q = mapping.get_mapping(FieldRole.QUERY)
    r = mapping.get_mapping(FieldRole.RESPONSE)
    assert q is not None and q.json_path == "question"
    assert r is not None and r.json_path == "answer"


def test_get_default_mapping_none_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("nlpeep.user_config.user_config_path", lambda: tmp_path / "nope.toml")
    assert get_default_mapping() is None


def test_get_default_mapping_none_when_no_mapping_key(tmp_path, monkeypatch):
    cfg_path = tmp_path / "nlpeep" / "config.toml"
    monkeypatch.setattr("nlpeep.user_config.user_config_path", lambda: cfg_path)
    save_user_config({"defaults": {"display": {"theme": "dark"}}})
    assert get_default_mapping() is None


# -- macOS XDG override --------------------------------------------------------


def test_macos_uses_dot_config_not_library(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("HOME", "/Users/tester")
    p = user_config_path()
    assert ".config" in str(p)
    assert "Library" not in str(p)


# -- llm section round-trip ----------------------------------------------------


def test_llm_section_round_trips(tmp_path, monkeypatch):
    cfg_path = tmp_path / "nlpeep" / "config.toml"
    monkeypatch.setattr("nlpeep.user_config.user_config_path", lambda: cfg_path)

    config = {
        "llm": {
            "model": "openai/gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 1024,
        },
    }
    save_user_config(config)
    loaded = load_user_config()
    assert loaded["llm"] == config["llm"]
    assert loaded["llm"]["temperature"] == 0.0
    assert loaded["llm"]["max_tokens"] == 1024
