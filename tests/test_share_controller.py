"""Tests for GUI share controller helpers."""

from __future__ import annotations

from typing import Any

import pytest

from core.sharing import ShareProviderInfo, ShareResult
from gui.share_controller import ShareController


class _DummyProvider:
    def __init__(self) -> None:
        self.info = ShareProviderInfo(
            name="dummy",
            label="DummyShare",
            description="Test provider",
        )

    def share(self, payload: str, prompt: Any | None = None) -> ShareResult:
        return ShareResult(
            provider=self.info,
            url="https://example.com/share",
            payload_chars=len(payload),
        )


class _DummyUsageLogger:
    def log_share(self, *, provider: str, prompt_name: str, payload_chars: int) -> None:  # noqa: D401
        return None


class _DummyIndicator:
    def __init__(self, *_: Any, **__: Any) -> None:
        return None

    def run(self, func, *args):  # noqa: ANN001
        return func(*args)


class _DummyClipboard:
    def __init__(self) -> None:
        self.text: str | None = None

    def setText(self, text: str) -> None:  # noqa: N802 - Qt style API
        self.text = text


@pytest.fixture(autouse=True)
def _patch_processing_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gui.share_controller.ProcessingIndicator", _DummyIndicator)


def test_share_controller_opens_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the default browser automatically when preference is enabled."""

    opened: list[str] = []
    monkeypatch.setattr(
        "gui.share_controller.webbrowser.open",
        lambda url, new=2: opened.append(url) or True,
    )
    clipboard = _DummyClipboard()
    monkeypatch.setattr("PySide6.QtGui.QGuiApplication.clipboard", lambda: clipboard)

    controller = ShareController(
        None,
        toast_callback=lambda *_: None,
        status_callback=lambda *_: None,
        error_callback=lambda *_: None,
        usage_logger=_DummyUsageLogger(),
        preference_supplier=lambda: True,
    )
    controller.register_provider(_DummyProvider())

    assert controller.share_payload(
        "dummy",
        "payload",
        prompt_name="Test",
        indicator_title="Share",
        error_title="Error",
    )
    assert clipboard.text == "https://example.com/share"
    assert opened == ["https://example.com/share"]


def test_share_controller_respects_disabled_auto_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip opening the browser when preference function returns False."""

    monkeypatch.setattr("gui.share_controller.webbrowser.open", lambda *_, **__: True)
    monkeypatch.setattr("PySide6.QtGui.QGuiApplication.clipboard", lambda: _DummyClipboard())

    controller = ShareController(
        None,
        toast_callback=lambda *_: None,
        status_callback=lambda *_: None,
        error_callback=lambda *_: None,
        usage_logger=_DummyUsageLogger(),
        preference_supplier=lambda: False,
    )
    controller.register_provider(_DummyProvider())

    controller.share_payload(
        "dummy",
        "payload",
        prompt_name="Test",
        indicator_title="Share",
        error_title="Error",
    )
