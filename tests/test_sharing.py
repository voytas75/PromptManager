"""Tests for prompt sharing helpers.

Updates:
  v0.1.0 - 2025-12-07 - Cover shared footer helper for prompts and results.
"""

from __future__ import annotations

from pytest import MonkeyPatch

from core.sharing import append_share_footer


def test_append_share_footer_appends_metadata_block(monkeypatch: MonkeyPatch) -> None:
    """Ensure the footer includes attribution and the injected date."""

    monkeypatch.setattr("core.sharing._current_share_date", lambda: "2025-12-25")

    payload = append_share_footer("Result body")

    assert payload == (
        "Result body\n\n---\nPromptManager | Author: https://github.com/voytas75 | Shared: 2025-12-25"
    )


def test_append_share_footer_strips_trailing_whitespace(monkeypatch: MonkeyPatch) -> None:
    """Remove trailing spaces before adding the footer block."""

    monkeypatch.setattr("core.sharing._current_share_date", lambda: "2025-12-31")

    payload = append_share_footer("Result body   \n")

    assert payload.startswith("Result body\n\n---\n")
    assert payload.endswith(
        "PromptManager | Author: https://github.com/voytas75 | Shared: 2025-12-31"
    )
