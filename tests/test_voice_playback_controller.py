"""Tests for the LiteLLM voice playback controller."""

from __future__ import annotations

from typing import Any, cast

import pytest

import gui.voice_playback_controller as voice_module
from gui.voice_playback_controller import VoicePlaybackController, VoicePlaybackError


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def connect(self, callback: Any) -> None:
        self._callbacks.append(callback)


class _FakeAudioOutput:
    def __init__(self, _parent: object | None = None) -> None:
        self.parent = _parent


class _FakePlayer:
    def __init__(self, _parent: object | None = None) -> None:
        self.parent = _parent
        self.audio_output: object | None = None
        self.playbackStateChanged = _FakeSignal()

    def setAudioOutput(self, output: object) -> None:
        self.audio_output = output


@pytest.fixture
def _fake_multimedia(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(voice_module, "_MULTIMEDIA_AVAILABLE", True)
    monkeypatch.setattr(voice_module, "QMediaPlayer", _FakePlayer)
    monkeypatch.setattr(voice_module, "QAudioOutput", _FakeAudioOutput)


def test_voice_playback_requires_multimedia_backend() -> None:
    controller = VoicePlaybackController()
    if controller.is_supported:
        pytest.skip("Qt multimedia is available; this test targets the fallback path.")
    with pytest.raises(VoicePlaybackError, match="Qt multimedia backend"):
        controller.play_text(
            "Hello",
            {
                "litellm_tts_model": "openai/tts-1",
                "litellm_api_key": "test-key",
            },
        )


def test_voice_playback_requires_configured_model_when_supported() -> None:
    controller = VoicePlaybackController()
    if not controller.is_supported:
        pytest.skip("Qt multimedia unavailable; cannot verify configuration validation.")
    with pytest.raises(VoicePlaybackError, match="LiteLLM TTS model"):
        controller.play_text("Test", {"litellm_api_key": "test-key"})


def test_voice_playback_does_not_create_multimedia_backend_on_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_module, "_multimedia_available", True)
    monkeypatch.setattr(voice_module, "QMediaPlayer", _FakePlayer)
    monkeypatch.setattr(voice_module, "QAudioOutput", _FakeAudioOutput)
    controller = VoicePlaybackController()

    assert controller.is_supported is True
    assert cast("Any", controller)._player is None
    assert cast("Any", controller)._audio_output is None


def test_voice_playback_creates_backend_on_first_play_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_module, "_multimedia_available", True)
    monkeypatch.setattr(voice_module, "QMediaPlayer", _FakePlayer)
    monkeypatch.setattr(voice_module, "QAudioOutput", _FakeAudioOutput)
    controller = VoicePlaybackController()

    with pytest.raises(VoicePlaybackError, match="LiteLLM TTS model"):
        controller.play_text("Test", {"litellm_api_key": "test-key"})

    assert isinstance(cast("Any", controller)._player, _FakePlayer)
    assert isinstance(cast("Any", controller)._audio_output, _FakeAudioOutput)
    assert cast("Any", controller)._player.audio_output is cast("Any", controller)._audio_output
