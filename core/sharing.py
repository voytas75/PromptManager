"""Prompt sharing helpers for external paste services.

Updates:
  v0.1.7 - 2025-12-12 - Retry transient network failures for share providers.
  v0.1.6 - 2025-12-07 - Add Rentry provider and management-note support.
  v0.1.5 - 2025-12-07 - Add PrivateBin provider for zero-knowledge sharing.
  v0.1.4 - 2025-12-07 - Provide shared footer helper for prompts and results.
  v0.1.3 - 2025-12-05 - Sort imports for lint compliance.
  v0.1.2 - 2025-12-04 - Append footer metadata with app name, author link, and share date.
  v0.1.1 - 2025-11-30 - Document ShareText provider methods for lint compliance.
  v0.1.0 - 2025-11-28 - Add ShareText provider and prompt formatting helper.
"""

from __future__ import annotations

import base64
import json
import secrets
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from core.exceptions import ShareProviderError
from core.retry import is_retryable_httpx_error, is_retryable_url_error, retry

if TYPE_CHECKING:
    from models.prompt_model import Prompt

_APP_NAME = "PromptManager"
_APP_AUTHOR_URL = "https://github.com/voytas75"
_BASE58_ALPHABET: Final[str] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_PRIVATEBIN_ITERATIONS: Final[int] = 150_000
_PRIVATEBIN_KEY_BYTES: Final[int] = 32
_PRIVATEBIN_SALT_BYTES: Final[int] = 8
_PRIVATEBIN_IV_BYTES: Final[int] = 16
_PRIVATEBIN_TAG_BITS: Final[int] = 128
_PRIVATEBIN_ALLOWED_EXPIRATIONS: Final[set[str]] = {
    "5min",
    "10min",
    "1hour",
    "1day",
    "1week",
    "1month",
    "1year",
    "never",
}
_PRIVATEBIN_ALLOWED_FORMATS: Final[set[str]] = {
    "plaintext",
    "syntaxhighlighting",
    "markdown",
}
_PRIVATEBIN_ALLOWED_COMPRESSION: Final[set[str]] = {"zlib", "none"}
_DEFAULT_PRIVATEBIN_BASE = "https://privatebin.net/"
_DEFAULT_PRIVATEBIN_EXPIRATION = "1week"
_DEFAULT_PRIVATEBIN_FORMAT = "markdown"
_DEFAULT_PRIVATEBIN_COMPRESSION = "zlib"
_DEFAULT_RENTRY_BASE = "https://rentry.co"


def _current_share_date() -> str:
    """Return the ISO-8601 date string used in footer metadata."""
    return date.today().isoformat()


def append_share_footer(payload: str) -> str:
    """Append the standard share footer block to *payload* text."""
    base_text = (payload or "").rstrip()
    footer_line = f"{_APP_NAME} | Author: {_APP_AUTHOR_URL} | Shared: {_current_share_date()}"
    if not base_text:
        return f"---\n{footer_line}"
    return f"{base_text}\n\n---\n{footer_line}"


def _raw_deflate(data: bytes, compression: str) -> bytes:
    if compression == "none":
        return data
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _compact_json_bytes(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _base58_encode(data: bytes) -> str:
    if not data:
        return _BASE58_ALPHABET[0]
    num = int.from_bytes(data, "big")
    encoded: list[str] = []
    while num > 0:
        num, remainder = divmod(num, 58)
        encoded.append(_BASE58_ALPHABET[remainder])
    encoded_str = "".join(reversed(encoded)) or _BASE58_ALPHABET[0]
    leading_zeroes = 0
    for byte in data:
        if byte == 0:
            leading_zeroes += 1
        else:
            break
    return f"{_BASE58_ALPHABET[0] * leading_zeroes}{encoded_str}"


def _normalise_privatebin_base_url(base_url: str) -> str:
    candidate = base_url.strip()
    if not candidate:
        raise ValueError("PrivateBin base URL must not be empty.")
    parts = urlsplit(candidate)
    if not parts.scheme or not parts.netloc:
        raise ValueError("PrivateBin base URL must include a scheme and hostname.")
    path = parts.path or "/"
    if not path.endswith("/"):
        path = f"{path}/"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _normalise_rentry_base_url(base_url: str) -> str:
    candidate = base_url.strip().rstrip("/")
    if not candidate:
        raise ValueError("Rentry base URL must not be empty.")
    parts = urlsplit(candidate)
    if not parts.scheme or not parts.netloc:
        raise ValueError("Rentry base URL must include a scheme and hostname.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True, slots=True)
class ShareProviderInfo:
    """Metadata describing a share provider entry."""

    name: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class ShareResult:
    """Details returned after a share succeeds."""

    provider: ShareProviderInfo
    url: str
    payload_chars: int
    delete_url: str | None = None
    management_note: str | None = None


class ShareProvider(Protocol):
    """Protocol implemented by share providers."""

    info: ShareProviderInfo

    def share(
        self,
        payload: str,
        prompt: Prompt | None = None,
    ) -> ShareResult:  # pragma: no cover - Protocol
        """Share *payload* (optionally describing *prompt*) and return a :class:`ShareResult`."""
        ...


def format_prompt_for_share(
    prompt: Prompt,
    *,
    include_description: bool = True,
    include_scenarios: bool = True,
    include_examples: bool = True,
    include_metadata: bool = True,
) -> str:
    """Return a readable text payload for uploading to sharing services."""
    lines: list[str] = []
    title = prompt.name.strip() if prompt.name else "Untitled prompt"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Category: {prompt.category or 'Uncategorised'}")
    language = (prompt.language or "en").strip() or "en"
    lines.append(f"Language: {language}")
    if prompt.tags:
        tags = ", ".join(sorted(str(tag).strip() for tag in prompt.tags if str(tag).strip()))
        if tags:
            lines.append(f"Tags: {tags}")
    if prompt.quality_score is not None and prompt.rating_count > 0:
        lines.append(f"Quality: {prompt.quality_score:.1f}/10 ({prompt.rating_count} ratings)")
    lines.append("")
    if include_description:
        lines.append("## Description")
        description_text = prompt.description or "No description provided."
        lines.append(description_text)
        lines.append("")
    lines.append("## Prompt Body")
    lines.append(prompt.context or "No prompt text provided.")
    lines.append("")
    if include_scenarios and prompt.scenarios:
        lines.append("## Scenarios")
        for scenario in prompt.scenarios:
            scenario_text = str(scenario).strip()
            if scenario_text:
                lines.append(f"- {scenario_text}")
        lines.append("")
    if include_examples:
        example_sections: list[str] = []
        if prompt.example_input:
            example_sections.append(f"Example input:\n{prompt.example_input}")
        if prompt.example_output:
            example_sections.append(f"Example output:\n{prompt.example_output}")
        if example_sections:
            lines.append("## Examples")
            lines.append("\n\n".join(example_sections))
            lines.append("")
    if include_metadata:
        metadata = prompt.to_metadata()
        lines.append("## Metadata")
        lines.append(json.dumps(metadata, ensure_ascii=False, indent=2))
    payload = "\n".join(lines).strip()
    payload = payload or "Prompt content unavailable."
    return append_share_footer(payload)


class ShareTextProvider:
    """Share prompts via https://sharetext.io."""

    _API_URL = "https://sharetext.io/api/text"
    _SITE_URL = "https://sharetext.io"
    _USER_AGENT = "PromptManager/PromptShare"

    def __init__(self, *, expiry: str = "1M", timeout: float = 15.0) -> None:
        """Configure ShareText client defaults such as paste expiry and timeout."""
        self._expiry = expiry
        self._timeout = timeout
        self.info = ShareProviderInfo(
            name="sharetext",
            label="ShareText",
            description="Publish prompts via sharetext.io and copy the link to the clipboard.",
        )

    def share(self, payload: str, prompt: Prompt | None = None) -> ShareResult:
        """Upload *payload* to ShareText and return the share metadata."""
        body = json.dumps(
            {"text": payload, "expiry": self._expiry},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self._API_URL,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self._USER_AGENT,
            },
        )
        try:

            def _send_request() -> bytes:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return response.read()

            response_body = retry(
                _send_request,
                should_retry=is_retryable_url_error,
            )
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            reason = getattr(exc, "reason", str(exc))
            raise ShareProviderError(f"Unable to reach ShareText: {reason}") from exc
        try:
            data = json.loads(response_body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - propagate parsing errors
            raise ShareProviderError("ShareText returned an invalid response.") from exc
        slug = str(data.get("slug", "")).strip()
        if not slug:
            raise ShareProviderError("ShareText response was missing a share identifier.")
        share_url = f"{self._SITE_URL}/{slug}"
        delete_key = str(data.get("deleteKey", "")).strip() or None
        delete_url = None
        if delete_key:
            delete_url = f"{self._SITE_URL}/api/delete?slug={slug}&key={delete_key}"
        return ShareResult(
            provider=self.info,
            url=share_url,
            payload_chars=len(payload),
            delete_url=delete_url,
        )


class PrivateBinProvider:
    """Share prompts via a PrivateBin instance (https://privatebin.net/)."""

    _USER_AGENT = "PromptManager/PrivateBinShare"

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_PRIVATEBIN_BASE,
        expiration: str = _DEFAULT_PRIVATEBIN_EXPIRATION,
        formatter: str = _DEFAULT_PRIVATEBIN_FORMAT,
        compression: str = _DEFAULT_PRIVATEBIN_COMPRESSION,
        burn_after_reading: bool = False,
        open_discussion: bool = False,
        confirm_before_open: bool = False,
        timeout: float = 20.0,
    ) -> None:
        """Configure connection + encryption defaults for a PrivateBin instance."""
        try:
            normalised_base = _normalise_privatebin_base_url(base_url)
        except ValueError as exc:
            raise ShareProviderError(str(exc)) from exc
        expiration_value = expiration.strip().lower()
        if expiration_value not in _PRIVATEBIN_ALLOWED_EXPIRATIONS:
            raise ShareProviderError(
                f"Unsupported PrivateBin expiration '{expiration}'. Allowed: "
                f"{', '.join(sorted(_PRIVATEBIN_ALLOWED_EXPIRATIONS))}."
            )
        formatter_value = formatter.strip().lower()
        if formatter_value not in _PRIVATEBIN_ALLOWED_FORMATS:
            raise ShareProviderError(
                f"Unsupported PrivateBin formatter '{formatter}'. Allowed: "
                f"{', '.join(sorted(_PRIVATEBIN_ALLOWED_FORMATS))}."
            )
        compression_value = compression.strip().lower()
        if compression_value not in _PRIVATEBIN_ALLOWED_COMPRESSION:
            raise ShareProviderError(
                f"Unsupported PrivateBin compression '{compression}'. Allowed: "
                f"{', '.join(sorted(_PRIVATEBIN_ALLOWED_COMPRESSION))}."
            )
        self._base_url = normalised_base
        self._expiration = expiration_value
        self._formatter = formatter_value
        self._compression = compression_value
        self._burn_after_reading = burn_after_reading
        self._open_discussion = open_discussion
        self._timeout = timeout
        self._fragment_prefix = "-" if confirm_before_open else ""
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Requested-With": "JSONHttpRequest",
            "User-Agent": self._USER_AGENT,
        }
        self.info = ShareProviderInfo(
            name="privatebin",
            label="PrivateBin",
            description="Publish encrypted prompts via your configured PrivateBin instance.",
        )

    def share(self, payload: str, prompt: Prompt | None = None) -> ShareResult:
        """Encrypt *payload* and upload it to the configured PrivateBin instance."""
        plaintext_bytes = _compact_json_bytes({"paste": payload})
        message_bytes = _raw_deflate(plaintext_bytes, self._compression)
        secret_key = secrets.token_bytes(_PRIVATEBIN_KEY_BYTES)
        salt = secrets.token_bytes(_PRIVATEBIN_SALT_BYTES)
        iv = secrets.token_bytes(_PRIVATEBIN_IV_BYTES)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=_PRIVATEBIN_KEY_BYTES,
            salt=salt,
            iterations=_PRIVATEBIN_ITERATIONS,
        )
        derived_key = kdf.derive(secret_key)
        adata = [
            [
                base64.b64encode(iv).decode("utf-8"),
                base64.b64encode(salt).decode("utf-8"),
                _PRIVATEBIN_ITERATIONS,
                _PRIVATEBIN_KEY_BYTES * 8,
                _PRIVATEBIN_TAG_BITS,
                "aes",
                "gcm",
                self._compression,
            ],
            self._formatter,
            1 if self._open_discussion else 0,
            1 if self._burn_after_reading else 0,
        ]
        aad_bytes = _compact_json_bytes(adata)
        cipher = Cipher(
            algorithms.AES(derived_key),
            modes.GCM(iv, None, _PRIVATEBIN_TAG_BITS // 8),
        )
        encryptor = cipher.encryptor()
        encryptor.authenticate_additional_data(aad_bytes)
        ciphertext = encryptor.update(message_bytes) + encryptor.finalize()
        payload_dict = {
            "v": 2,
            "adata": adata,
            "ct": base64.b64encode(ciphertext + encryptor.tag).decode("utf-8"),
            "meta": {"expire": self._expiration},
        }
        request_body = _compact_json_bytes(payload_dict)
        try:

            def _send_request() -> httpx.Response:
                response = httpx.post(
                    self._base_url,
                    content=request_body,
                    headers=self._headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response

            response = retry(_send_request, should_retry=is_retryable_httpx_error)
        except httpx.HTTPError as exc:
            raise ShareProviderError(f"Unable to reach PrivateBin: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise ShareProviderError("PrivateBin returned an invalid response.") from exc
        status = int(data.get("status", 1))
        if status != 0:
            message = str(data.get("message") or "Unknown error.")
            raise ShareProviderError(f"PrivateBin rejected the share: {message}")
        paste_id = str(data.get("id", "")).strip()
        if not paste_id:
            raise ShareProviderError("PrivateBin response was missing a paste identifier.")
        delete_token = str(data.get("deletetoken", "")).strip() or None
        fragment = _base58_encode(secret_key)
        if self._fragment_prefix:
            fragment = f"{self._fragment_prefix}{fragment}"
        share_url = f"{self._base_url}?{paste_id}#{fragment}"
        delete_url = None
        if delete_token:
            delete_url = f"{self._base_url}?pasteid={paste_id}&deletetoken={delete_token}"
        return ShareResult(
            provider=self.info,
            url=share_url,
            payload_chars=len(payload),
            delete_url=delete_url,
        )


class RentryProvider:
    """Share prompts via https://rentry.co markdown pastes."""

    _USER_AGENT = "PromptManager/RentryShare"

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_RENTRY_BASE,
        timeout: float = 15.0,
        metadata: str | None = None,
    ) -> None:
        """Configure a Rentry client with optional metadata overrides."""
        try:
            normalised_base = _normalise_rentry_base_url(base_url)
        except ValueError as exc:
            raise ShareProviderError(str(exc)) from exc
        self._base_url = normalised_base
        self._timeout = timeout
        self._metadata = metadata.strip() if metadata and metadata.strip() else None
        referer = normalised_base if normalised_base.endswith("/") else f"{normalised_base}/"
        self._headers = {
            "User-Agent": self._USER_AGENT,
            "Referer": referer,
        }
        self.info = ShareProviderInfo(
            name="rentry",
            label="Rentry",
            description="Publish markdown-friendly prompts to rentry.co with edit codes.",
        )

    def share(self, payload: str, prompt: Prompt | None = None) -> ShareResult:
        """Upload *payload* to Rentry and capture the edit code for later updates."""
        stripped_payload = payload.strip()
        if not stripped_payload:
            raise ShareProviderError("Share payload is empty.")
        try:
            with httpx.Client(
                headers=self._headers,
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                csrf_token = retry(
                    lambda: self._fetch_csrf_token(client),
                    should_retry=is_retryable_httpx_error,
                )

                def _send_request() -> httpx.Response:
                    response = client.post(
                        f"{self._base_url}/api/new",
                        data=self._build_request_body(csrf_token, stripped_payload),
                    )
                    response.raise_for_status()
                    return response

                response = retry(_send_request, should_retry=is_retryable_httpx_error)
        except httpx.HTTPError as exc:
            raise ShareProviderError(f"Unable to reach Rentry: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - unexpected type
            raise ShareProviderError("Rentry returned an invalid response.") from exc
        status_text = str(data.get("status", "")).strip().lower()
        if status_text not in {"200", "ok", "success"}:
            message = str(data.get("content") or data.get("errors") or "Unknown error.").strip()
            raise ShareProviderError(f"Rentry rejected the share: {message}")
        share_url = str(data.get("url") or "").strip()
        slug = str(data.get("url_short") or "").strip()
        if not share_url and slug:
            share_url = f"{self._base_url.rstrip('/')}/{slug}"
        if not share_url:
            raise ShareProviderError("Rentry response was missing a share URL.")
        edit_code = str(data.get("edit_code") or "").strip()
        management_note = None
        if edit_code and slug:
            management_note = (
                f"Edit this entry via {self._base_url.rstrip('/')}/{slug}/edit "
                f"using code: {edit_code}"
            )
        elif edit_code:
            management_note = f"Store this Rentry edit code for updates: {edit_code}"
        return ShareResult(
            provider=self.info,
            url=share_url,
            payload_chars=len(stripped_payload),
            management_note=management_note,
        )

    def _fetch_csrf_token(self, client: httpx.Client) -> str:
        response = client.get(self._base_url)
        response.raise_for_status()
        token = client.cookies.get("csrftoken")
        if not token:
            raise ShareProviderError("Rentry did not provide a CSRF token.")
        return token

    def _build_request_body(self, token: str, payload: str) -> dict[str, str]:
        body = {
            "csrfmiddlewaretoken": token,
            "text": payload,
        }
        if self._metadata:
            body["metadata"] = self._metadata
        return body


__all__ = [
    "append_share_footer",
    "ShareProvider",
    "ShareProviderInfo",
    "ShareResult",
    "ShareTextProvider",
    "PrivateBinProvider",
    "RentryProvider",
    "format_prompt_for_share",
]
