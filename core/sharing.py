"""Prompt sharing helpers for external paste services.

Updates:
  v0.1.2 - 2025-12-04 - Append footer metadata with app name, author link, and share date.
  v0.1.1 - 2025-11-30 - Document ShareText provider methods for lint compliance.
  v0.1.0 - 2025-11-28 - Add ShareText provider and prompt formatting helper.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from core.exceptions import ShareProviderError

if TYPE_CHECKING:
    from models.prompt_model import Prompt

_APP_NAME = "PromptManager"
_APP_AUTHOR_URL = "https://github.com/voytas75"


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
    footer_date = date.today().isoformat()
    lines.append("")
    lines.append("---")
    lines.append(f"{_APP_NAME} | Author: {_APP_AUTHOR_URL} | Shared: {footer_date}")
    payload = "\n".join(lines).strip()
    return payload or "Prompt content unavailable."


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
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                response_body = response.read()
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


__all__ = [
    "ShareProvider",
    "ShareProviderInfo",
    "ShareResult",
    "ShareTextProvider",
    "format_prompt_for_share",
]
