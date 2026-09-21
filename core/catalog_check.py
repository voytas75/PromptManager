"""Provider-free integrity checks for PromptManager catalog assets."""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from core.templating import TemplateRenderer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from models.prompt_chain_model import PromptChain
    from models.prompt_model import Prompt


@dataclass(frozen=True, slots=True)
class CatalogCheckIssue:
    """One deterministic catalog-integrity finding."""

    code: str
    severity: str
    message: str
    prompt_ids: tuple[str, ...] = ()
    chain_id: str | None = None

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-ready issue payload."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CatalogCheckReport:
    """Read-only catalog-check result with deterministic issue ordering."""

    prompt_count: int
    chain_count: int
    issues: tuple[CatalogCheckIssue, ...]

    @property
    def error_count(self) -> int:
        """Return the number of error-severity issues."""
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Return the number of warning-severity issues."""
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-ready report payload."""
        return {
            "summary": {
                "prompts": self.prompt_count,
                "chains": self.chain_count,
                "errors": self.error_count,
                "warnings": self.warning_count,
            },
            "issues": [issue.to_record() for issue in self.issues],
        }


def run_catalog_check(
    prompts: Sequence[Prompt],
    chains: Sequence[PromptChain],
    *,
    renderer: TemplateRenderer | None = None,
) -> CatalogCheckReport:
    """Check prompt and chain integrity without providers or persistence changes."""
    template_renderer = renderer or TemplateRenderer()
    prompt_list = list(prompts)
    chain_list = list(chains)
    prompt_ids = {prompt.id for prompt in prompt_list}
    issues: list[CatalogCheckIssue] = []

    _check_duplicate_names(prompt_list, issues)
    _check_duplicate_bodies(prompt_list, issues)
    _check_template_syntax(prompt_list, template_renderer, issues)
    _check_related_prompt_references(prompt_list, prompt_ids, issues)
    _check_missing_embeddings(prompt_list, issues)
    _check_chain_prompt_references(chain_list, prompt_ids, issues)

    return CatalogCheckReport(
        prompt_count=len(prompt_list),
        chain_count=len(chain_list),
        issues=tuple(sorted(issues, key=_issue_sort_key)),
    )


def _check_duplicate_names(prompts: Sequence[Prompt], issues: list[CatalogCheckIssue]) -> None:
    groups: dict[str, list[Prompt]] = defaultdict(list)
    for prompt in prompts:
        name = str(prompt.name or "").strip()
        if name:
            groups[name].append(prompt)
    for name, group in sorted(groups.items()):
        if len(group) > 1:
            issues.append(
                CatalogCheckIssue(
                    code="CAT001",
                    severity="warning",
                    message=f"Duplicate exact prompt name: {name!r}.",
                    prompt_ids=tuple(sorted(str(prompt.id) for prompt in group)),
                )
            )


def _normalise_body(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _check_duplicate_bodies(prompts: Sequence[Prompt], issues: list[CatalogCheckIssue]) -> None:
    groups: dict[str, list[Prompt]] = defaultdict(list)
    for prompt in prompts:
        body = _normalise_body(prompt.context)
        if body:
            groups[body].append(prompt)
    for group in groups.values():
        if len(group) > 1:
            issues.append(
                CatalogCheckIssue(
                    code="CAT002",
                    severity="warning",
                    message="Duplicate normalized prompt body.",
                    prompt_ids=tuple(sorted(str(prompt.id) for prompt in group)),
                )
            )


def _check_template_syntax(
    prompts: Sequence[Prompt],
    renderer: TemplateRenderer,
    issues: list[CatalogCheckIssue],
) -> None:
    for prompt in prompts:
        body = str(prompt.context or "")
        if not body.strip():
            continue
        try:
            renderer.extract_variables(body)
        except Exception as exc:
            issues.append(
                CatalogCheckIssue(
                    code="CAT003",
                    severity="error",
                    message=f"Invalid template syntax: {exc}",
                    prompt_ids=(str(prompt.id),),
                )
            )


def _check_related_prompt_references(
    prompts: Sequence[Prompt],
    prompt_ids: set[uuid.UUID],
    issues: list[CatalogCheckIssue],
) -> None:
    for prompt in prompts:
        for raw_reference in prompt.related_prompts:
            try:
                reference_id = uuid.UUID(str(raw_reference))
            except (TypeError, ValueError):
                reason = f"is not a UUID: {raw_reference!r}"
            else:
                if reference_id in prompt_ids:
                    continue
                reason = f"points to missing prompt {reference_id}"
            issues.append(
                CatalogCheckIssue(
                    code="CAT004",
                    severity="error",
                    message=f"Related prompt reference {reason}.",
                    prompt_ids=(str(prompt.id),),
                )
            )


def _check_missing_embeddings(prompts: Sequence[Prompt], issues: list[CatalogCheckIssue]) -> None:
    missing = tuple(sorted(str(prompt.id) for prompt in prompts if prompt.ext4 is None))
    if missing:
        issues.append(
            CatalogCheckIssue(
                code="CAT005",
                severity="warning",
                message=f"{len(missing)} prompt(s) have no stored embedding vector.",
                prompt_ids=missing,
            )
        )


def _check_chain_prompt_references(
    chains: Sequence[PromptChain],
    prompt_ids: set[uuid.UUID],
    issues: list[CatalogCheckIssue],
) -> None:
    for chain in sorted(chains, key=lambda item: str(item.id)):
        missing = tuple(
            sorted(str(step.prompt_id) for step in chain.steps if step.prompt_id not in prompt_ids)
        )
        if missing:
            issues.append(
                CatalogCheckIssue(
                    code="CAT006",
                    severity="error",
                    message=(
                        f"Chain {chain.name!r} references {len(missing)} missing prompt step(s)."
                    ),
                    prompt_ids=missing,
                    chain_id=str(chain.id),
                )
            )


def _issue_sort_key(issue: CatalogCheckIssue) -> tuple[str, str, tuple[str, ...], str]:
    return issue.code, issue.chain_id or "", issue.prompt_ids, issue.message


__all__ = ["CatalogCheckIssue", "CatalogCheckReport", "run_catalog_check"]
