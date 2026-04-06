"""Tests for optional TTS module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from nlpeep import tts
from nlpeep.tts import _resolve_env, is_available

# -- is_available --------------------------------------------------------------


def test_is_available_false_without_litellm():
    """Returns False when litellm is not importable."""
    with patch.dict(sys.modules, {"litellm": None}):
        assert is_available() is False


def test_is_available_false_without_playsound3():
    """Returns False when playsound3 is not importable."""
    with patch.dict(sys.modules, {"playsound3": None}):
        assert is_available() is False


def test_is_available_true_with_deps():
    """Returns True when both deps are importable."""
    with patch.dict(sys.modules, {"litellm": MagicMock(), "playsound3": MagicMock()}):
        assert is_available() is True


# -- _resolve_env --------------------------------------------------------------


def test_resolve_env_reads_var(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
    assert _resolve_env("os.environ/TEST_API_KEY") == "sk-test-123"


def test_resolve_env_missing_var_returns_empty(monkeypatch):
    monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
    assert _resolve_env("os.environ/NONEXISTENT_VAR_12345") == ""


def test_resolve_env_passthrough_string():
    assert _resolve_env("plain_value") == "plain_value"


def test_resolve_env_passthrough_non_string():
    assert _resolve_env(42) == 42
    assert _resolve_env(1.5) == 1.5


# -- _get_tts_config -----------------------------------------------------------


def test_get_tts_config_resolves_env(monkeypatch):
    """Config values with os.environ/ prefix are resolved at load time."""
    monkeypatch.setenv("MY_TTS_KEY", "sk-resolved")
    monkeypatch.setattr(
        "nlpeep.user_config.load_user_config",
        lambda: {"tts": {"api_key": "os.environ/MY_TTS_KEY", "model": "openai/tts-1"}},
    )
    config = tts._get_tts_config()
    assert config["api_key"] == "sk-resolved"
    assert config["model"] == "openai/tts-1"


def test_get_tts_config_empty_when_no_section(monkeypatch):
    monkeypatch.setattr("nlpeep.user_config.load_user_config", lambda: {})
    assert tts._get_tts_config() == {}


# -- speak / stop lifecycle ---------------------------------------------------


def test_speak_calls_litellm_and_plays(monkeypatch):
    """speak() generates audio via litellm and plays with playsound3 blocking."""
    monkeypatch.setattr(
        tts,
        "_get_tts_config",
        lambda: {"model": "openai/tts-1", "voice": "nova", "speed": 1.25},
    )

    mock_response = MagicMock()
    mock_litellm = MagicMock()
    mock_litellm.speech.return_value = mock_response

    mock_playsound_fn = MagicMock()
    mock_ps3 = MagicMock()
    mock_ps3.playsound = mock_playsound_fn

    with patch.dict(sys.modules, {"litellm": mock_litellm, "playsound3": mock_ps3}):
        tts.speak("Hello world")

        # litellm.speech called with correct args
        mock_litellm.speech.assert_called_once()
        call_kw = mock_litellm.speech.call_args.kwargs
        assert call_kw["model"] == "openai/tts-1"
        assert call_kw["voice"] == "nova"
        assert call_kw["input"] == "Hello world"
        assert call_kw["speed"] == 1.25

        # Audio saved to temp file
        mock_response.stream_to_file.assert_called_once()

        # playsound called blocking so the worker thread waits for completion
        mock_playsound_fn.assert_called_once()
        assert mock_playsound_fn.call_args.kwargs.get("block") is True

        # State cleared automatically when speak() returns (finally block)
        assert not tts.is_speaking()


def test_stop_when_not_speaking():
    """stop() is safe to call when nothing is playing."""
    tts._audio_file = None
    mock_ps3 = MagicMock()
    with patch.dict(sys.modules, {"playsound3": mock_ps3}):
        tts.stop()
    assert not tts.is_speaking()


def test_speak_without_optional_config(monkeypatch):
    """speak() uses defaults when config has no api_key/api_base/speed."""
    monkeypatch.setattr(tts, "_get_tts_config", lambda: {})

    mock_response = MagicMock()
    mock_litellm = MagicMock()
    mock_litellm.speech.return_value = mock_response

    mock_ps3 = MagicMock()

    with patch.dict(sys.modules, {"litellm": mock_litellm, "playsound3": mock_ps3}):
        tts.speak("test")

        call_kw = mock_litellm.speech.call_args.kwargs
        assert call_kw["model"] == "openai/tts-1"
        assert call_kw["voice"] == "alloy"
        assert "api_key" not in call_kw
        assert "api_base" not in call_kw
        assert "speed" not in call_kw
