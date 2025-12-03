"""Tests for the LiteLLM voice playback controller."""

from __future__ import annotations

import pytest

from gui.voice_playback_controller import VoicePlaybackController, VoicePlaybackError


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
