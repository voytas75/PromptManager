"""Centralised share-provider orchestration for prompt and result payloads.

Updates:
  v0.1.0 - 2025-11-30 - Introduce ShareController to manage provider menus and execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QMenu, QWidget

from core.exceptions import ShareProviderError

from .processing_indicator import ProcessingIndicator

if TYPE_CHECKING:
    from core.sharing import ShareProvider

    from .usage_logger import IntentUsageLogger
else:  # pragma: no cover - fallback stub for runtime annotations
    IntentUsageLogger = object  # type: ignore[assignment]


class ShareController(QObject):
    """Manage share providers, provider selection menus, and share execution."""

    def __init__(
        self,
        parent: QWidget,
        *,
        toast_callback: Callable[[str], None],
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
        usage_logger: IntentUsageLogger,
    ) -> None:
        """Initialise share helpers and callbacks required for feedback."""
        super().__init__(parent)
        self._parent = parent
        self._toast_callback = self._ensure_callable(toast_callback, "toast_callback")
        self._status_callback = self._ensure_callable(status_callback, "status_callback")
        self._error_callback = self._ensure_callable(error_callback, "error_callback")
        self._usage_logger = usage_logger
        self._providers: dict[str, ShareProvider] = {}

    def register_provider(self, provider: ShareProvider) -> None:
        """Store a share provider so it can be offered to users."""
        self._providers[provider.info.name] = provider

    def has_providers(self) -> bool:
        """Return True when at least one share provider is configured."""
        return bool(self._providers)

    def choose_provider(self, anchor: QWidget | None) -> str | None:
        """Display a provider picker next to *anchor* and return the selection."""
        if not self._providers:
            self._error_callback("Sharing unavailable", "No share providers are configured.")
            return None
        if anchor is None:
            self._error_callback("Sharing unavailable", "Provider anchor is missing.")
            return None
        menu = QMenu(self._parent)
        for provider in self._providers.values():
            action = menu.addAction(provider.info.label)
            action.setData(provider.info.name)
            action.setToolTip(provider.info.description)
        # QMenu.exec_ synchronously returns the chosen QAction (see
        # https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QMenu).
        chosen = menu.exec_(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        if chosen is None:
            return None
        return str(chosen.data())

    def share_payload(
        self,
        provider_name: str,
        payload: str,
        *,
        prompt_name: str | None,
        indicator_title: str,
        error_title: str,
        prompt: object | None = None,
    ) -> bool:
        """Share *payload* using *provider_name* and log the outbound metadata."""
        provider = self._providers.get(provider_name)
        if provider is None:
            self._error_callback(
                "Sharing unavailable",
                "Selected share provider is not registered.",
            )
            return False
        payload_text = payload or ""
        if not payload_text.strip():
            self._error_callback(error_title, "Share payload is empty.")
            return False
        indicator = ProcessingIndicator(
            self._parent,
            f"Sharing via {provider.info.label}…",
            title=indicator_title,
        )
        try:
            result = indicator.run(provider.share, payload_text, prompt)
        except ShareProviderError as exc:
            self._error_callback(error_title, str(exc))
            return False
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(result.url)
        message = f"{result.provider.label} link copied to clipboard."
        self._toast_callback(message)
        self._status_callback(message, 5000)
        prompt_label = prompt_name or getattr(prompt, "name", "Result Output")
        self._usage_logger.log_share(
            provider=result.provider.name,
            prompt_name=prompt_label,
            payload_chars=result.payload_chars,
        )
        if result.delete_url:
            self._status_callback(
                f"Delete this share later via: {result.delete_url}",
                10000,
            )
        return True

    @staticmethod
    def _ensure_callable(
        callback: Callable[..., Any],
        name: str,
    ) -> Callable[..., Any]:
        """Validate that ``callback`` is callable, raising otherwise."""
        if not isinstance(callback, Callable):
            raise TypeError(f"{name} must be callable")
        return callback


__all__ = ["ShareController"]
