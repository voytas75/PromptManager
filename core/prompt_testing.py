"""Provider-free fixture runner for deterministic prompt-template tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast

from core.templating import TemplateRenderer

if TYPE_CHECKING:
    from models.prompt_model import Prompt


class PromptTestSuiteError(ValueError):
    """Raised when a prompt-test suite does not meet the v1 fixture contract."""


@dataclass(frozen=True, slots=True)
class PromptTestCase:
    """One exact-output template fixture."""

    id: str
    variables: dict[str, object]
    expected: str


@dataclass(frozen=True, slots=True)
class PromptTestCaseResult:
    """One provider-free fixture outcome without exposing prompt bodies."""

    id: str
    status: str
    error: str | None = None

    def to_record(self) -> dict[str, str | None]:
        """Return a JSON-ready case result."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptTestReport:
    """Deterministic test-suite report for one selected prompt."""

    prompt_id: str
    prompt_name: str
    suite_path: str
    case_count: int
    cases: tuple[PromptTestCaseResult, ...]

    @property
    def passed_count(self) -> int:
        """Return number of passing cases."""
        return sum(case.status == "passed" for case in self.cases)

    @property
    def failed_count(self) -> int:
        """Return number of failed cases."""
        return sum(case.status == "failed" for case in self.cases)

    @property
    def ok(self) -> bool:
        """Return whether every test case passed."""
        return self.failed_count == 0

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-ready report payload."""
        return {
            "ok": self.ok,
            "prompt": {"id": self.prompt_id, "name": self.prompt_name},
            "suite": {"path": self.suite_path, "case_count": self.case_count},
            "summary": {
                "total": self.case_count,
                "passed": self.passed_count,
                "failed": self.failed_count,
            },
            "cases": [case.to_record() for case in self.cases],
        }


def parse_prompt_test_suite(payload: object) -> tuple[PromptTestCase, ...]:
    """Parse and validate the v1 JSON suite fixture contract."""
    if not isinstance(payload, Mapping):
        raise PromptTestSuiteError("Suite must be a JSON object with a non-empty 'cases' list.")
    raw_payload = cast("Mapping[str, object]", payload)
    raw_cases = raw_payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise PromptTestSuiteError("Suite must include a non-empty 'cases' list.")
    typed_cases = cast("list[object]", raw_cases)

    cases: list[PromptTestCase] = []
    seen_ids: set[str] = set()
    for raw_case in typed_cases:
        if not isinstance(raw_case, Mapping):
            raise PromptTestSuiteError("Suite cases must be JSON objects.")
        case = cast("Mapping[str, object]", raw_case)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise PromptTestSuiteError("Suite cases must include a non-empty string field 'id'.")
        normalized_id = case_id.strip()
        if normalized_id in seen_ids:
            raise PromptTestSuiteError(f"Suite case IDs must be unique: {normalized_id!r}.")
        seen_ids.add(normalized_id)
        expected = case.get("expected")
        if not isinstance(expected, str):
            raise PromptTestSuiteError(
                f"Suite case {normalized_id!r} must include string field 'expected'."
            )
        raw_variables = case.get("variables", {})
        if not isinstance(raw_variables, Mapping):
            raise PromptTestSuiteError(
                f"Suite case {normalized_id!r} field 'variables' must be a JSON object."
            )
        typed_variables = cast("Mapping[object, object]", raw_variables)
        variables = {str(key): value for key, value in typed_variables.items()}
        cases.append(PromptTestCase(id=normalized_id, variables=variables, expected=expected))
    return tuple(cases)


def run_prompt_test_suite(
    prompt: Prompt,
    cases: Sequence[PromptTestCase],
    *,
    suite_path: str,
    renderer: TemplateRenderer | None = None,
) -> PromptTestReport:
    """Run exact-output local template fixtures without calling a provider."""
    template_renderer = renderer or TemplateRenderer()
    template_text = str(prompt.context or "")
    results: list[PromptTestCaseResult] = []
    for case in cases:
        try:
            result = template_renderer.render(template_text, case.variables)
        except Exception as exc:  # pragma: no cover - defensive renderer boundary
            results.append(
                PromptTestCaseResult(case.id, "failed", f"Template rendering failed: {exc}")
            )
            continue
        if result.errors:
            results.append(PromptTestCaseResult(case.id, "failed", result.errors[0]))
        elif result.rendered_text != case.expected:
            results.append(
                PromptTestCaseResult(case.id, "failed", "Rendered output did not match expected.")
            )
        else:
            results.append(PromptTestCaseResult(case.id, "passed"))
    return PromptTestReport(
        prompt_id=str(prompt.id),
        prompt_name=str(prompt.name),
        suite_path=suite_path,
        case_count=len(cases),
        cases=tuple(results),
    )


def failed_suite_report(
    prompt: Prompt,
    *,
    suite_path: str,
    error: str,
) -> PromptTestReport:
    """Return a deterministic report for a suite-level validation failure."""
    return PromptTestReport(
        prompt_id=str(prompt.id),
        prompt_name=str(prompt.name),
        suite_path=suite_path,
        case_count=0,
        cases=(PromptTestCaseResult("suite", "failed", error),),
    )


__all__ = [
    "PromptTestCase",
    "PromptTestCaseResult",
    "PromptTestReport",
    "PromptTestSuiteError",
    "failed_suite_report",
    "parse_prompt_test_suite",
    "run_prompt_test_suite",
]
