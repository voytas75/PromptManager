"""Provider-free technical validation for one persisted prompt."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from core.templating import TemplateRenderer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from models.prompt_model import Prompt


@dataclass(frozen=True, slots=True)
class PromptValidationIssue:
    """One deterministic technical validation finding."""

    code: str
    severity: str
    message: str

    def to_record(self) -> dict[str, str]:
        """Return a JSON-ready issue payload."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptValidationReport:
    """Provider-free validation result for one prompt asset."""

    prompt_id: str
    prompt_name: str
    variables: tuple[str, ...]
    issues: tuple[PromptValidationIssue, ...]

    @property
    def error_count(self) -> int:
        """Return the number of error-severity findings."""
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Return the number of warning-severity findings."""
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def valid(self) -> bool:
        """Return whether the report has no error-severity findings."""
        return self.error_count == 0

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-ready report payload."""
        return {
            "valid": self.valid,
            "prompt": {"id": self.prompt_id, "name": self.prompt_name},
            "variables": list(self.variables),
            "summary": {"errors": self.error_count, "warnings": self.warning_count},
            "issues": [issue.to_record() for issue in self.issues],
        }


def validate_prompt(
    prompt: Prompt,
    known_prompt_ids: Iterable[uuid.UUID],
    *,
    renderer: TemplateRenderer | None = None,
) -> PromptValidationReport:
    """Validate local prompt structure and references without rendering or mutation."""
    template_renderer = renderer or TemplateRenderer()
    known_ids = set(known_prompt_ids)
    issues: list[PromptValidationIssue] = []
    variables: tuple[str, ...] = ()

    if not str(prompt.name or "").strip():
        issues.append(PromptValidationIssue("VAL001", "error", "Prompt name is blank."))
    if not str(prompt.description or "").strip():
        issues.append(PromptValidationIssue("VAL002", "error", "Prompt description is blank."))

    body = str(prompt.context or "")
    if not body.strip():
        issues.append(
            PromptValidationIssue("VAL003", "warning", "Prompt body is empty or whitespace-only.")
        )
    else:
        try:
            variables = tuple(template_renderer.extract_variables(body))
        except Exception as exc:
            issues.append(
                PromptValidationIssue("VAL004", "error", f"Invalid template syntax: {exc}")
            )

    issues.extend(_validate_related_prompts(prompt, known_ids))
    if _tags_need_attention(prompt.tags):
        issues.append(
            PromptValidationIssue(
                "VAL006",
                "warning",
                "Tags contain blank values or duplicate values case-insensitively.",
            )
        )

    return PromptValidationReport(
        prompt_id=str(prompt.id),
        prompt_name=str(prompt.name or ""),
        variables=variables,
        issues=tuple(sorted(issues, key=lambda issue: (issue.code, issue.message))),
    )


def _validate_related_prompts(
    prompt: Prompt,
    known_ids: set[uuid.UUID],
) -> list[PromptValidationIssue]:
    issues: list[PromptValidationIssue] = []
    for raw_reference in prompt.related_prompts:
        try:
            reference_id = uuid.UUID(str(raw_reference))
        except (TypeError, ValueError):
            message = f"Related prompt reference is not a UUID: {raw_reference!r}."
        else:
            if reference_id in known_ids:
                continue
            message = f"Related prompt reference points to missing prompt {reference_id}."
        issues.append(PromptValidationIssue("VAL005", "error", message))
    return issues


def _tags_need_attention(tags: Iterable[object]) -> bool:
    seen: set[str] = set()
    for raw_tag in tags:
        tag = str(raw_tag).strip()
        if not tag:
            return True
        normalized = tag.casefold()
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


__all__ = ["PromptValidationIssue", "PromptValidationReport", "validate_prompt"]
