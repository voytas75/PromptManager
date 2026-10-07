"""Tests for the LiteLLM voice playback controller."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

import gui.voice_playback_controller as voice_module
from gui.voice_playback_controller import VoicePlaybackController, VoicePlaybackError

_RUNTIME = {
    "model": "azure/gpt-audio-1.5",
    "voice": "alloy",
    "api_key": "private-key",
    "api_base": "https://private-endpoint",
    "api_version": "private-version",
}


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


def _audio_response(data: object = "SUQzYXVkaW8=") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(audio=SimpleNamespace(data=data)))]
    )


@pytest.mark.parametrize(
    "model", ["azure/gpt-audio-1.5", "openai/gpt-4o-audio-preview", "gpt-4o-mini-audio-preview"]
)
def test_audio_chat_route_preserves_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, model: str
) -> None:
    provider = SimpleNamespace(speech=Mock(), completion=Mock(return_value=_audio_response()))
    monkeypatch.setattr(voice_module, "litellm", provider)
    controller = VoicePlaybackController()
    response = cast("Any", controller)._request_audio("Exact text?", {**_RUNTIME, "model": model})
    assert response.content == base64.b64decode("SUQzYXVkaW8=")
    provider.speech.assert_not_called()
    kwargs = provider.completion.call_args.kwargs
    assert kwargs["model"] == model
    for key in ("api_key", "api_base", "api_version"):
        assert kwargs[key] == _RUNTIME[key]
    assert kwargs["modalities"] == ["audio", "text"]
    assert kwargs["audio"] == {"voice": "alloy", "format": "mp3"}
    assert kwargs["messages"][1]["content"] == "Exact text?"
    assert "exactly" in kwargs["messages"][0]["content"]
    path = tmp_path / "audio.mp3"
    assert cast("Any", controller)._write_response_to_file(response, path, True) == (False, False)
    assert path.read_bytes() == response.content


@pytest.mark.parametrize("model", ["azure/tts-hd", "custom-deployment", "gpt-audio-not-a-family"])
def test_standard_and_unknown_models_keep_speech(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    provider = SimpleNamespace(
        speech=Mock(return_value=SimpleNamespace(content=b"audio")), completion=Mock()
    )
    monkeypatch.setattr(voice_module, "litellm", provider)
    cast("Any", VoicePlaybackController())._request_audio("text", {**_RUNTIME, "model": model})
    provider.speech.assert_called_once()
    assert provider.speech.call_args.kwargs["max_retries"] == 0
    assert provider.speech.call_args.kwargs["response_format"] == "mp3"
    provider.completion.assert_not_called()


@pytest.mark.parametrize(
    "model,first,second",
    [("custom-deployment", "speech", "completion"), ("azure/gpt-audio", "completion", "speech")],
)
def test_explicit_compatibility_error_adapts_once(
    monkeypatch: pytest.MonkeyPatch, caplog: Any, model: str, first: str, second: str
) -> None:
    provider = SimpleNamespace(
        speech=Mock(return_value=SimpleNamespace(content=b"audio")),
        completion=Mock(return_value=_audio_response()),
    )
    getattr(provider, first).side_effect = RuntimeError(
        "OperationNotSupported: model does not support this operation private-key"
    )
    monkeypatch.setattr(voice_module, "litellm", provider)
    cast("Any", VoicePlaybackController())._request_audio("text", {**_RUNTIME, "model": model})
    getattr(provider, first).assert_called_once()
    getattr(provider, second).assert_called_once()
    assert "operation" in caplog.text
    assert "private-key" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        "401 authentication OperationNotSupported",
        "429 rate limit OperationNotSupported",
        "timeout OperationNotSupported",
        "network connection OperationNotSupported",
        "DeploymentNotFound OperationNotSupported",
        "404 NotFound OperationNotSupported",
        "unsupported parameter voice",
        "unrelated failure",
    ],
)
def test_noncompatibility_errors_never_retry(monkeypatch: pytest.MonkeyPatch, error: str) -> None:
    provider = SimpleNamespace(speech=Mock(side_effect=RuntimeError(error)), completion=Mock())
    monkeypatch.setattr(voice_module, "litellm", provider)
    with pytest.raises(RuntimeError):
        cast("Any", VoicePlaybackController())._request_audio(
            "text", {**_RUNTIME, "model": "custom"}
        )
    provider.speech.assert_called_once()
    assert provider.speech.call_args.kwargs["max_retries"] == 0
    assert provider.speech.call_args.kwargs["response_format"] == "mp3"
    provider.completion.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        _audio_response(""),
        _audio_response("%%%"),
        _audio_response(None),
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())]),
    ],
)
def test_malformed_audio_does_not_retry(monkeypatch: pytest.MonkeyPatch, response: object) -> None:
    provider = SimpleNamespace(speech=Mock(), completion=Mock(return_value=response))
    monkeypatch.setattr(voice_module, "litellm", provider)
    with pytest.raises(VoicePlaybackError, match="audio"):
        cast("Any", VoicePlaybackController())._request_audio("text", _RUNTIME)
    provider.speech.assert_not_called()


def test_alternate_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(
        speech=Mock(side_effect=RuntimeError("OperationNotSupported")),
        completion=Mock(side_effect=RuntimeError("OperationNotSupported")),
    )
    monkeypatch.setattr(voice_module, "litellm", provider)
    with pytest.raises(RuntimeError):
        cast("Any", VoicePlaybackController())._request_audio(
            "text", {**_RUNTIME, "model": "custom"}
        )
    provider.speech.assert_called_once()
    provider.completion.assert_called_once()


def test_worker_failure_is_actionable_and_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(
        speech=Mock(
            side_effect=RuntimeError("private-key https://private-endpoint private-version")
        ),
        completion=Mock(),
    )
    monkeypatch.setattr(voice_module, "litellm", provider)
    controller = VoicePlaybackController()
    errors: list[str] = []
    controller.playback_failed.connect(errors.append)
    cast("Any", controller)._is_preparing = True
    cast("Any", controller)._download_and_prepare("text", {**_RUNTIME, "model": "custom"}, True)
    assert len(errors) == 1
    assert "Settings" in errors[0]
    assert "private" not in errors[0]
    assert controller.is_active is False


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_http_status_blocks_adaptation(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    error = RuntimeError("OperationNotSupported")
    cast("Any", error).status_code = status
    provider = SimpleNamespace(speech=Mock(side_effect=error), completion=Mock())
    monkeypatch.setattr(voice_module, "litellm", provider)
    with pytest.raises(RuntimeError):
        cast("Any", VoicePlaybackController())._request_audio(
            "text", {**_RUNTIME, "model": "custom"}
        )
    provider.completion.assert_not_called()


def test_cancelled_request_does_not_adapt(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(
        speech=Mock(side_effect=RuntimeError("OperationNotSupported")), completion=Mock()
    )
    monkeypatch.setattr(voice_module, "litellm", provider)
    controller = VoicePlaybackController()
    cast("Any", controller)._stop_event.set()
    with pytest.raises(RuntimeError):
        cast("Any", controller)._request_audio("text", {**_RUNTIME, "model": "custom"})
    provider.completion.assert_not_called()


def test_audio_chat_worker_readiness_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(speech=Mock(), completion=Mock(return_value=_audio_response()))
    monkeypatch.setattr(voice_module, "litellm", provider)
    controller = VoicePlaybackController()
    paths: list[str] = []
    controller.playback_ready.connect(paths.append)
    cast("Any", controller)._download_and_prepare("text", _RUNTIME, True)
    assert len(paths) == 1
    assert cast("Any", controller)._temp_path.read_bytes() == b"ID3audio"
    controller.stop()
    assert cast("Any", controller)._temp_path is None
