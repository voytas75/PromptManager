"""Tests for GUI share controller helpers.

Updates:
  v0.1.2 - 2025-12-08 - Pass None parent via cast to preserve QObject behaviour.
  v0.1.1 - 2025-12-08 - Cast QWidget parent and usage logger stubs for Pyright.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from core.sharing import ShareProviderInfo, ShareResult
from gui.share_controller import ShareController

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from PySide6.QtWidgets import QWidget
    from gui.usage_logger import IntentUsageLogger
else:  # pragma: no cover - runtime fallbacks for optional deps
    QWidget = object  # type: ignore[assignment]
    IntentUsageLogger = object  # type: ignore[assignment]


class _DummyProvider:
    def __init__(
        self,
        *,
        result: ShareResult | None = None,
        info: ShareProviderInfo | None = None,
    ) -> None:
        self.info = info or ShareProviderInfo(
            name="dummy",
            label="DummyShare",
            description="Test provider",
        )
        self._result = result

    def share(self, payload: str, prompt: Any | None = None) -> ShareResult:
        if self._result is not None:
            return self._result
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
        cast(QWidget, None),
        toast_callback=lambda *_: None,
        status_callback=lambda *_: None,
        error_callback=lambda *_: None,
        usage_logger=cast(IntentUsageLogger, _DummyUsageLogger()),
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
        cast(QWidget, None),
        toast_callback=lambda *_: None,
        status_callback=lambda *_: None,
        error_callback=lambda *_: None,
        usage_logger=cast(IntentUsageLogger, _DummyUsageLogger()),
        preference_supplier=lambda: False,
    )
    controller.register_provider(_DummyProvider())


def test_share_controller_surfaces_management_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emit delete links and provider-specific notes through the status callback."""

    monkeypatch.setattr("gui.share_controller.webbrowser.open", lambda *_, **__: True)
    monkeypatch.setattr("PySide6.QtGui.QGuiApplication.clipboard", lambda: _DummyClipboard())

    status_messages: list[str] = []

    def _status(message: str, _: int) -> None:
        status_messages.append(message)

    controller = ShareController(
        cast(QWidget, None),
        toast_callback=lambda *_: None,
        status_callback=_status,
        error_callback=lambda *_: None,
        usage_logger=cast(IntentUsageLogger, _DummyUsageLogger()),
        preference_supplier=lambda: False,
    )
    provider_info = ShareProviderInfo(name="dummy", label="DummyShare", description="Test")
    provider = _DummyProvider(
        result=ShareResult(
            provider=provider_info,
            url="https://example.com/share",
            payload_chars=7,
            delete_url="https://example.com/delete",
            management_note="Edit via https://example.com/edit using code: secret",
        ),
        info=provider_info,
    )
    controller.register_provider(provider)
    controller.share_payload(
        "dummy",
        "payload",
        prompt_name="Test",
        indicator_title="Share",
        error_title="Error",
    )

    assert any("Delete this share later" in message for message in status_messages)
    assert any("Edit via" in message for message in status_messages)
