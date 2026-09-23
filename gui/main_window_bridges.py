"""Bridges that safely delegate MainWindow events to handlers.

Updates:
  v0.16.1 - 2025-12-02 - Defer handler lookup so early-wired callbacks work reliably.
  v0.16.0 - 2025-12-02 - Introduce prompt, workspace, and template handler bridges.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .main_window_handlers import PromptActionsHandler


BridgeCallback = Callable[..., object | None]


class _BaseBridge:
    """Utility base that shields handler access when uninitialised."""

    def __init__(self, handler_supplier: Callable[[], object | None]) -> None:
        """Store the callable used to fetch the latest handler instance."""
        self._handler_supplier = handler_supplier

    def __getattr__(self, attribute: str) -> BridgeCallback:
        """Return the requested handler attribute, deferring lookup when needed."""
        handler = self._handler_supplier()
        if handler is None:

            def _deferred(*args: object, **kwargs: object) -> object | None:
                live_handler = self._handler_supplier()
                if live_handler is None:
                    return None
                callback = cast("BridgeCallback", getattr(live_handler, attribute))
                return callback(*args, **kwargs)

            return _deferred
        return cast("BridgeCallback", getattr(handler, attribute))


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

    def __getattr__(self, attribute: str) -> BridgeCallback:
        """Provide lazy attribute lookup with graceful close fallbacks."""
        if attribute == "close_application":

            def _close(*args: object, **kwargs: object) -> object | None:
                handler = self._handler_supplier()
                if handler is None:
                    self._close_fallback()
                    return None
                callback = cast("BridgeCallback", getattr(handler, attribute))
                return callback(*args, **kwargs)

            return _close
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
