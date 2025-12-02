"""Bridges that safely delegate MainWindow events to handlers.

Updates:
  v0.16.0 - 2025-12-02 - Introduce prompt, workspace, and template handler bridges.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable

    from .main_window_handlers import (
        PromptActionsHandler,
        TemplatePreviewHandler,
        WorkspaceInputHandler,
    )
else:  # pragma: no cover - runtime placeholders for type-only imports
    Callable = object  # type: ignore[assignment]
    PromptActionsHandler = object  # type: ignore[assignment]
    TemplatePreviewHandler = object  # type: ignore[assignment]
    WorkspaceInputHandler = object  # type: ignore[assignment]


class _BaseBridge:
    """Utility base that shields handler access when uninitialised."""

    def __init__(self, handler_supplier: Callable[[], object | None]) -> None:
        """Store the callable used to fetch the latest handler instance."""
        self._handler_supplier = handler_supplier

    def __getattr__(self, attribute: str):
        """Return the requested handler attribute, falling back to a no-op."""
        handler = self._handler_supplier()
        if handler is None:
            return lambda *args, **kwargs: None
        return getattr(handler, attribute)


class PromptActionsBridge(_BaseBridge):
    """Delegate prompt-centric UI events to :class:`PromptActionsHandler`."""

    def __init__(
        self,
        handler_supplier: Callable[[], PromptActionsHandler | None],
        *,
        close_fallback: Callable[[], None],
    ) -> None:
        """Initialise the bridge with a close fallback for shutdown events."""
        super().__init__(handler_supplier)
        self._close_fallback = close_fallback

    def __getattr__(self, attribute: str):
        """Provide lazy attribute lookup with graceful close fallbacks."""
        if attribute == "close_application":
            handler = self._handler_supplier()
            if handler is None:
                return lambda *args, **kwargs: self._close_fallback()
            return getattr(handler, attribute)
        return super().__getattr__(attribute)


class WorkspaceInputBridge(_BaseBridge):
    """Delegate workspace interactions to :class:`WorkspaceInputHandler`."""


class TemplatePreviewBridge(_BaseBridge):
    """Delegate template preview events to :class:`TemplatePreviewHandler`."""


__all__ = [
    "PromptActionsBridge",
    "TemplatePreviewBridge",
    "WorkspaceInputBridge",
]
