"""Tests for prompt sharing helpers.

Updates:
  v0.1.1 - 2025-12-07 - Cover PrivateBin provider payload construction and error handling.
  v0.1.0 - 2025-12-07 - Cover shared footer helper for prompts and results.
"""

from __future__ import annotations

import json

import pytest
from pytest import MonkeyPatch

from core.exceptions import ShareProviderError
from core.sharing import PrivateBinProvider, append_share_footer


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


def test_privatebin_provider_uploads_payload(monkeypatch: MonkeyPatch) -> None:
    """Encrypt payloads, send them to PrivateBin, and report share URLs."""

    captured: dict[str, object] = {}

    class _DummyResponse:
        def __init__(self) -> None:
            self.text = '{"status":0}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"status": 0, "id": "abc123", "deletetoken": "del-token"}

    def fake_post(
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> _DummyResponse:
        captured["url"] = url
        captured["body"] = content
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr("core.sharing.httpx.post", fake_post)
    monkeypatch.setattr("core.sharing.secrets.token_bytes", lambda size: bytes([size]) * size)
    monkeypatch.setattr("core.sharing._base58_encode", lambda _: "encoded-key")

    provider = PrivateBinProvider(
        base_url="https://bin.example/secure/",
        expiration="10min",
        formatter="plaintext",
        compression="none",
        burn_after_reading=True,
        open_discussion=True,
    )

    result = provider.share("Payload body", prompt=None)

    assert result.url == "https://bin.example/secure/?abc123#encoded-key"
    assert result.delete_url == "https://bin.example/secure/?pasteid=abc123&deletetoken=del-token"
    assert result.payload_chars == len("Payload body")

    payload = json.loads(captured["body"].decode("utf-8"))
    assert payload["meta"]["expire"] == "10min"
    assert payload["adata"][1] == "plaintext"
    assert payload["adata"][2] == 1
    assert payload["adata"][3] == 1
    headers = captured["headers"]
    assert headers["X-Requested-With"] == "JSONHttpRequest"


def test_privatebin_provider_handles_error(monkeypatch: MonkeyPatch) -> None:
    """Raise a ShareProviderError when PrivateBin rejects the request."""

    class _ErrorResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"status": 1, "message": "rate limited"}

    monkeypatch.setattr("core.sharing.httpx.post", lambda *_, **__: _ErrorResponse())

    provider = PrivateBinProvider()

    with pytest.raises(ShareProviderError) as excinfo:
        provider.share("payload", prompt=None)

    assert "rate limited" in str(excinfo.value)
