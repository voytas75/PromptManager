"""Provider-free advisory maintainability lint for one persisted prompt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from core.templating import TemplateRenderer

if TYPE_CHECKING:
    from models.prompt_model import Prompt


_ACTION_CUES = frozenset(
    {
        "analyze",
        "answer",
        "compare",
        "create",
        "describe",
        "draft",
        "explain",
        "generate",
        "identify",
        "list",
        "plan",
        "provide",
        "recommend",
        "review",
        "summarize",
        "translate",
        "write",
        "analizuj",
        "opisz",
        "podsumuj",
        "porównaj",
        "przygotuj",
        "stwórz",
        "utwórz",
        "wyjaśnij",
        "wygeneruj",
    }
)


@dataclass(frozen=True, slots=True)
class PromptLintIssue:
    """One deterministic advisory finding about a prompt asset."""

    code: str
    severity: str
    message: str

    def to_record(self) -> dict[str, str]:
        """Return a JSON-ready issue representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptLintReport:
    """Stable provider-free lint result for one prompt asset."""

    prompt_id: str
    prompt_name: str
    body_characters: int
    body_lines: int
    variables: tuple[str, ...]
    template_parse_error: str | None
    issues: tuple[PromptLintIssue, ...]

    @property
    def warning_count(self) -> int:
        """Return the number of advisory warnings."""
        return len(self.issues)

    @property
    def clean(self) -> bool:
        """Return whether no advisory findings were emitted."""
        return not self.issues

    def to_record(self) -> dict[str, object]:
        """Return the stable JSON-ready lint contract."""
        return {
            "clean": self.clean,
            "prompt": {"id": self.prompt_id, "name": self.prompt_name},
            "metrics": {
                "body_characters": self.body_characters,
                "body_lines": self.body_lines,
                "variables": list(self.variables),
                "template_parse_error": self.template_parse_error,
            },
            "summary": {"warnings": self.warning_count},
            "issues": [issue.to_record() for issue in self.issues],
        }


def lint_prompt(
    prompt: Prompt,
    *,
    renderer: TemplateRenderer | None = None,
) -> PromptLintReport:
    """Return deterministic advisory findings without rendering, providers, or mutation."""
    body = str(prompt.context or "")
    description = str(prompt.description or "").strip()
    nonempty_lines = tuple(line.strip() for line in body.splitlines() if line.strip())
    variables, template_parse_error = _extract_variables(body, renderer or TemplateRenderer())
    issues: list[PromptLintIssue] = []

    if description and len(description) < 20:
        issues.append(
            PromptLintIssue(
                "LINT001",
                "warning",
                (
                    "Description is shorter than 20 characters; "
                    "expand it so the prompt purpose is clear."
                ),
            )
        )
    if body.strip() and not _has_action_cue(body):
        issues.append(
            PromptLintIssue(
                "LINT002",
                "warning",
                "Prompt body has no recognized action cue; state the requested task explicitly.",
            )
        )
    if len(nonempty_lines) >= 12 and not _has_structure_marker(nonempty_lines):
        issues.append(
            PromptLintIssue(
                "LINT003",
                "warning",
                (
                    "Long prompt body has no heading or numbered-list marker; "
                    "add structure for review."
                ),
            )
        )
    if _has_repeated_line(nonempty_lines):
        issues.append(
            PromptLintIssue(
                "LINT004",
                "warning",
                (
                    "Prompt body repeats a non-empty instruction line; "
                    "consolidate it to reduce ambiguity."
                ),
            )
        )
    if variables and not _mentions_input_context(description):
        issues.append(
            PromptLintIssue(
                "LINT005",
                "warning",
                (
                    "Prompt uses template variables but its description does not mention "
                    "input or variables."
                ),
            )
        )

    return PromptLintReport(
        prompt_id=str(prompt.id),
        prompt_name=str(prompt.name or ""),
        body_characters=len(body),
        body_lines=len(nonempty_lines),
        variables=variables,
        template_parse_error=template_parse_error,
        issues=tuple(sorted(issues, key=lambda issue: (issue.code, issue.message))),
    )


def _extract_variables(
    body: str,
    renderer: TemplateRenderer,
) -> tuple[tuple[str, ...], str | None]:
    if not body.strip():
        return (), None
    try:
        return tuple(renderer.extract_variables(body)), None
    except Exception as exc:
        return (), str(exc)


def _has_action_cue(body: str) -> bool:
    words = {word.strip(".,:;!?()[]{}\"'").casefold() for word in body.split()}
    return bool(words & _ACTION_CUES)


def _has_structure_marker(lines: tuple[str, ...]) -> bool:
    return any(line.startswith("#") or line[:2].isdigit() and line[2:3] == "." for line in lines)


def _has_repeated_line(lines: tuple[str, ...]) -> bool:
    normalized = [" ".join(line.casefold().split()) for line in lines]
    return len(normalized) != len(set(normalized))


def _mentions_input_context(description: str) -> bool:
    words = {
        "input",
        "inputs",
        "variable",
        "variables",
        "parameter",
        "parameters",
        "dane",
        "wejście",
    }
    normalized = {word.strip(".,:;!?()[]{}\"'").casefold() for word in description.split()}
    return bool(normalized & words)


__all__ = ["PromptLintIssue", "PromptLintReport", "lint_prompt"]
