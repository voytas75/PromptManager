"""Tests for GUI bridge helpers.

Updates:
  v0.1.0 - 2025-12-02 - Cover deferred handler binding and close fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass

from gui.main_window_bridges import PromptActionsBridge


@dataclass
class _HandlerStub:
    delete_called: bool = False
    close_called: bool = False

    def delete_current_prompt(self) -> None:
        self.delete_called = True

    def close_application(self) -> None:  # pragma: no cover - exercised indirectly
        self.close_called = True


def test_prompt_actions_bridge_defers_handler_lookup_when_prebound() -> None:
    """Callbacks captured before handler initialisation should execute later."""
    handler_box: dict[str, _HandlerStub | None] = {"handler": None}
    bridge = PromptActionsBridge(lambda: handler_box["handler"], close_fallback=lambda: None)
    delete_callback = bridge.delete_current_prompt

    handler_box["handler"] = _HandlerStub()
    delete_callback()

    assert handler_box["handler"].delete_called is True  # type: ignore[union-attr]


def test_prompt_actions_bridge_close_application_falls_back_until_ready() -> None:
    """Close request falls back while the handler is missing and delegates later."""
    handler_box: dict[str, _HandlerStub | None] = {"handler": None}
    fallback_called: list[bool] = []

    def _fallback() -> None:
        fallback_called.append(True)

    bridge = PromptActionsBridge(lambda: handler_box["handler"], close_fallback=_fallback)
    close_callback = bridge.close_application

    close_callback()
    assert fallback_called == [True]

    handler_box["handler"] = _HandlerStub()
    close_callback()

    assert handler_box["handler"].close_called is True  # type: ignore[union-attr]
