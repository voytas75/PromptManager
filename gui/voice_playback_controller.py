"""LiteLLM-powered voice playback for workspace results.

Updates:
  v0.1.0 - 2025-12-03 - Introduce controller that streams LiteLLM TTS output to Qt audio.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Mapping

from PySide6.QtCore import QObject, QUrl, Signal

_LITELLM_IMPORT_ERROR: str | None = None

try:  # pragma: no cover - optional dependency import
    import litellm
    try:
        from litellm.exceptions import LiteLLMException  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - missing attribute variations
        LiteLLMException = RuntimeError  # type: ignore[assignment]
        _LITELLM_IMPORT_ERROR = str(exc)
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    litellm = None  # type: ignore[assignment]
    LiteLLMException = RuntimeError  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - surface actual import failures
    litellm = None  # type: ignore[assignment]
    LiteLLMException = RuntimeError  # type: ignore[assignment]
    _LITELLM_IMPORT_ERROR = str(exc)

try:  # pragma: no cover - depends on optional Qt plugins
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except Exception:  # pragma: no cover - Qt multimedia missing
    QAudioOutput = None  # type: ignore[assignment]
    QMediaPlayer = None  # type: ignore[assignment]
    _MULTIMEDIA_AVAILABLE = False
else:  # pragma: no cover - exercised in GUI runtime
    _MULTIMEDIA_AVAILABLE = True

DEFAULT_TTS_VOICE = "alloy"


class VoicePlaybackError(RuntimeError):
    """Raised when LiteLLM voice playback cannot proceed."""


class VoicePlaybackController(QObject):
    """Manage LiteLLM TTS downloads and Qt audio playback."""

    playback_preparing = Signal()
    playback_started = Signal()
    playback_finished = Signal()
    playback_failed = Signal(str)
    playback_ready = Signal(str)

    def __init__(self, *, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._supported = bool(_MULTIMEDIA_AVAILABLE and QMediaPlayer and QAudioOutput)
        self._player: QMediaPlayer | None = None
        self._audio_output: QAudioOutput | None = None
        if self._supported and QMediaPlayer is not None and QAudioOutput is not None:
            self._player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_output)
            self._player.playbackStateChanged.connect(self._handle_state_changed)  # type: ignore[arg-type]
            self.playback_ready.connect(self._handle_playback_ready)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._temp_path: Path | None = None
        self._is_preparing = False
        self._is_playing = False

    @property
    def is_supported(self) -> bool:
        """Return ``True`` when Qt multimedia backends are available."""

        return self._supported

    @property
    def is_active(self) -> bool:
        """Return ``True`` while audio is preparing or playing."""

        return self._is_preparing or self._is_playing

    def play_text(
        self,
        text: str,
        runtime: Mapping[str, object | None],
        *,
        voice: str | None = None,
        stream_audio: bool = True,
    ) -> None:
        """Start LiteLLM text-to-speech playback for *text*."""

        if not self._supported:
            raise VoicePlaybackError("Qt multimedia backend is unavailable on this system.")
        if self._is_preparing or self._is_playing:
            raise VoicePlaybackError("Voice playback is already in progress.")
        cleaned = text.strip()
        if not cleaned:
            raise VoicePlaybackError("No prompt result is available to read aloud.")
        tts_model = runtime.get("litellm_tts_model")
        if not isinstance(tts_model, str) or not tts_model.strip():
            raise VoicePlaybackError("Set a LiteLLM TTS model in Settings before using voice playback.")
        api_key = runtime.get("litellm_api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise VoicePlaybackError("LiteLLM API key is required for voice playback.")

        api_base = runtime.get("litellm_api_base") if isinstance(runtime.get("litellm_api_base"), str) else None
        api_version = (
            runtime.get("litellm_api_version") if isinstance(runtime.get("litellm_api_version"), str) else None
        )
        self._is_preparing = True
        self.playback_preparing.emit()
        self._stop_event.clear()
        runtime_payload = {
            "model": tts_model.strip(),
            "voice": voice or DEFAULT_TTS_VOICE,
            "api_key": api_key.strip(),
            "api_base": api_base.strip() if api_base else None,
            "api_version": api_version.strip() if api_version else None,
        }
        self._worker = threading.Thread(
            target=self._download_and_prepare,
            args=(cleaned, runtime_payload, stream_audio),
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        """Stop playback or cancel any in-flight preparation."""

        self._stop_event.set()
        if self._player is not None and self._player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self._player.stop()
        else:
            self._finalise_stop()
            self.playback_finished.emit()

    def _download_and_prepare(
        self,
        text: str,
        runtime_payload: Mapping[str, str | None],
        stream_audio: bool,
    ) -> None:
        if litellm is None:
            message = _LITELLM_IMPORT_ERROR or (
                "LiteLLM is not installed; install litellm to enable voice playback."
            )
            self.playback_failed.emit(message)
            self._is_preparing = False
            return

        try:
            response = litellm.speech(  # type: ignore[attr-defined]
                model=runtime_payload.get("model"),
                voice=runtime_payload.get("voice"),
                input=text,
                api_key=runtime_payload.get("api_key"),
                api_base=runtime_payload.get("api_base"),
                api_version=runtime_payload.get("api_version"),
            )
        except LiteLLMException as exc:  # pragma: no cover - requires API access
            self.playback_failed.emit(f"LiteLLM TTS failed: {exc}")
            self._is_preparing = False
            return
        except Exception as exc:  # pragma: no cover - network/runtime errors
            self.playback_failed.emit(f"Voice playback failed: {exc}")
            self._is_preparing = False
            return

        fd, tmp_path = tempfile.mkstemp(prefix="prompt_manager_tts_", suffix=".mp3")
        os.close(fd)
        path = Path(tmp_path)
        self._temp_path = path
        try:
            started, interrupted = self._write_response_to_file(response, path, stream_audio)
        except VoicePlaybackError as exc:
            path.unlink(missing_ok=True)
            self._temp_path = None
            self.playback_failed.emit(str(exc))
            self._is_preparing = False
            return
        except LiteLLMException as exc:  # pragma: no cover - requires API access
            path.unlink(missing_ok=True)
            self._temp_path = None
            self.playback_failed.emit(f"LiteLLM TTS failed: {exc}")
            self._is_preparing = False
            return
        except Exception as exc:  # pragma: no cover - network/runtime errors
            path.unlink(missing_ok=True)
            self._temp_path = None
            self.playback_failed.emit(f"Voice playback failed: {exc}")
            self._is_preparing = False
            return

        if interrupted:
            path.unlink(missing_ok=True)
            self._temp_path = None
            self._is_preparing = False
            return

        if not started:
            self.playback_ready.emit(str(path))

    def _handle_playback_ready(self, path_str: str) -> None:
        if self._player is None:
            self.playback_failed.emit("Qt multimedia backend is unavailable on this system.")
            self._is_preparing = False
            return
        self._player.setSource(QUrl.fromLocalFile(path_str))
        self._player.play()
        self._is_preparing = False
        self._is_playing = True
        self.playback_started.emit()

    def _handle_state_changed(self, state) -> None:  # pragma: no cover - Qt signal
        if state == QMediaPlayer.PlaybackState.StoppedState and self._is_playing:
            self._finalise_stop()
            self.playback_finished.emit()

    def _write_response_to_file(
        self,
        response: object,
        path: Path,
        stream_audio: bool,
    ) -> tuple[bool, bool]:
        iterator_factory = getattr(response, "iter_bytes", None)
        if not callable(iterator_factory):
            stream_to_file = getattr(response, "stream_to_file", None)
            if callable(stream_to_file):
                stream_to_file(path)
            else:
                content = getattr(response, "content", None)
                if content is None and callable(getattr(response, "read", None)):
                    content = response.read()
                if content is None:
                    raise VoicePlaybackError("LiteLLM response did not expose audio content.")
                path.write_bytes(content)
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = 0
            if size == 0:
                raise VoicePlaybackError("LiteLLM returned an empty audio response.")
            return False, False

        started = False
        interrupted = False
        with path.open("wb") as handle:
            for chunk in iterator_factory():
                if self._stop_event.is_set():
                    interrupted = True
                    break
                if not chunk:
                    continue
                handle.write(chunk)
                handle.flush()
                if stream_audio and not started:
                    self.playback_ready.emit(str(path))
                    started = True

        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size == 0 and not interrupted:
            raise VoicePlaybackError("LiteLLM returned an empty audio response.")
        return started, interrupted

    def _finalise_stop(self) -> None:
        self._is_playing = False
        self._is_preparing = False
        if self._temp_path is not None:
            if self._player is not None:
                self._player.setSource(QUrl())
            try:
                self._temp_path.unlink()
            except PermissionError:
                cleanup_timer = threading.Timer(1.0, self._delayed_cleanup, args=(self._temp_path,))
                cleanup_timer.daemon = True
                cleanup_timer.start()
            self._temp_path = None
        self._stop_event.clear()

    @staticmethod
    def _delayed_cleanup(path: Path) -> None:  # pragma: no cover - timing dependent
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass


__all__ = ["VoicePlaybackController", "VoicePlaybackError"]
