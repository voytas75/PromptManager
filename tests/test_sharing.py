"""Tests for prompt sharing helpers.

Updates:
  v0.1.3 - 2025-12-12 - Cover retry behaviour for transient share provider failures.
  v0.1.2 - 2025-12-10 - Cover prompt formatting helper and PrivateBin URL validation.
  v0.1.1 - 2025-12-07 - Cover PrivateBin provider payload construction and error handling.
  v0.1.0 - 2025-12-07 - Cover shared footer helper for prompts and results.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pytest import MonkeyPatch

from core.exceptions import ShareProviderError
from core.sharing import (
    PrivateBinProvider,
    RentryProvider,
    _normalise_privatebin_base_url,
    append_share_footer,
    format_prompt_for_share,
)


class _DummyHttpxResponse:
    def __init__(self, data: dict[str, object] | None = None) -> None:
        self._data = data or {}

    def raise_for_status(self) -> None:  # noqa: D401 - tiny helper
        return None

    def json(self) -> dict[str, object]:
        return self._data


def test_append_share_footer_appends_metadata_block(monkeypatch: MonkeyPatch) -> None:
    """Ensure the footer includes attribution and the injected date."""

    monkeypatch.setattr("core.sharing._current_share_date", lambda: "2025-12-25")

    payload = append_share_footer("Result body")

    assert payload == (
        "Result body\n\n---\nPromptManager | Author: https://github.com/voytas75 | "
        "Shared: 2025-12-25"
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

    body = captured["body"]
    assert isinstance(body, bytes)
    payload = json.loads(body.decode("utf-8"))
    assert payload["meta"]["expire"] == "10min"
    assert payload["adata"][1] == "plaintext"
    assert payload["adata"][2] == 1
    assert payload["adata"][3] == 1
    headers = captured["headers"]
    assert isinstance(headers, dict)
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


def test_privatebin_provider_retries_transient_http_failures(monkeypatch: MonkeyPatch) -> None:
    """Retry transient HTTP failures when posting to PrivateBin."""
    calls: list[int] = []

    class _DummyResponse:
        def __init__(self, status_code: int, payload: dict[str, object] | None = None) -> None:
            self._status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            if self._status_code < 400:
                return None
            request = httpx.Request("POST", "https://bin.example/secure/")
            response = httpx.Response(self._status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(*_: object, **__: object) -> _DummyResponse:
        calls.append(1)
        if len(calls) == 1:
            return _DummyResponse(503)
        return _DummyResponse(200, {"status": 0, "id": "abc123", "deletetoken": "del-token"})

    monkeypatch.setattr("core.retry.time.sleep", lambda _: None)
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

    assert len(calls) == 2
    assert result.url == "https://bin.example/secure/?abc123#encoded-key"


def test_rentry_provider_creates_entry(monkeypatch: MonkeyPatch) -> None:
    """Upload payloads to Rentry and expose the edit-code management note."""

    captured: dict[str, object] = {}

    class _DummyClient:
        def __init__(self, **_: object) -> None:
            self.cookies = httpx.Cookies()
            self._csrf_seeded = False

        def __enter__(self) -> _DummyClient:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> _DummyHttpxResponse:
            captured["get_url"] = url
            if not self._csrf_seeded:
                self.cookies.set("csrftoken", "token-123", domain="rentry.co", path="/")
                self._csrf_seeded = True
            return _DummyHttpxResponse()

        def post(self, url: str, data: dict[str, str] | None = None) -> _DummyHttpxResponse:
            captured["post_url"] = url
            captured["post_data"] = data or {}
            return _DummyHttpxResponse(
                {
                    "status": "200",
                    "url": "https://rentry.co/demo",
                    "url_short": "demo",
                    "edit_code": "edit-secret",
                }
            )

    monkeypatch.setattr("core.sharing.httpx.Client", _DummyClient)

    provider = RentryProvider(metadata="OPTION_DISABLE_VIEWS = true")

    result = provider.share("Payload body", prompt=None)

    assert result.url == "https://rentry.co/demo"
    assert result.management_note is not None
    assert "edit-secret" in result.management_note
    assert captured["post_url"] == "https://rentry.co/api/new"
    post_data = captured["post_data"]
    assert isinstance(post_data, dict)
    assert post_data["csrfmiddlewaretoken"] == "token-123"
    assert post_data["metadata"] == "OPTION_DISABLE_VIEWS = true"


def test_rentry_provider_requires_csrf(monkeypatch: MonkeyPatch) -> None:
    """Raise a ShareProviderError when no CSRF cookie is returned."""

    class _TokenlessClient:
        def __init__(self, **_: object) -> None:
            self.cookies = httpx.Cookies()

        def __enter__(self) -> _TokenlessClient:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> _DummyHttpxResponse:
            return _DummyHttpxResponse()

        def post(self, url: str, data: dict[str, str] | None = None) -> _DummyHttpxResponse:
            return _DummyHttpxResponse()

    monkeypatch.setattr("core.sharing.httpx.Client", _TokenlessClient)

    provider = RentryProvider()

    with pytest.raises(ShareProviderError) as excinfo:
        provider.share("text", prompt=None)

    assert "CSRF" in str(excinfo.value)


def test_format_prompt_for_share_includes_sections(monkeypatch: MonkeyPatch) -> None:
    """Ensure prompt payload includes all sections and the footer."""

    import uuid

    from models.prompt_model import Prompt

    monkeypatch.setattr("core.sharing._current_share_date", lambda: "2025-12-31")

    prompt = Prompt(
        id=uuid.uuid4(),
        name="Shareable Prompt",
        description="Describe the behaviour",
        category="demo",
        tags=["tag-b", "tag-a"],
        language="pl",
        context="Do the thing",
        example_input="input sample",
        example_output="output sample",
        scenarios=[" first ", "second"],
    )

    payload = format_prompt_for_share(prompt)

    assert payload.startswith("# Shareable Prompt\n")
    assert "Category: demo" in payload
    assert "Language: pl" in payload
    assert "tag-a, tag-b" in payload  # tags sorted and trimmed
    assert "## Description" in payload and "Describe the behaviour" in payload
    assert "## Prompt Body" in payload and "Do the thing" in payload
    assert "## Examples" in payload and "input sample" in payload
    assert "output sample" in payload
    assert "## Scenarios" in payload and "- first" in payload and "- second" in payload
    assert payload.rstrip().endswith(
        "---\nPromptManager | Author: https://github.com/voytas75 | Shared: 2025-12-31"
    )


def test_normalise_privatebin_base_url_adds_trailing_slash() -> None:
    """Normalise base URLs and reject missing schemes."""

    normalised = _normalise_privatebin_base_url("https://pb.local/nested")
    assert normalised == "https://pb.local/nested/"

    with pytest.raises(ValueError):
        _normalise_privatebin_base_url("pb.local")
