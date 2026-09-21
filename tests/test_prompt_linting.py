"""Tests for deterministic provider-free prompt linting."""

from __future__ import annotations

import uuid

from core.prompt_linting import lint_prompt
from models.prompt_model import Prompt


def test_lint_prompt_reports_stable_advisory_findings() -> None:
    """Flag only documented deterministic quality cues without making them blocking."""
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Incident response",
        description="Help",
        category="Operations",
        context="""The incident is {{ incident_id }}.
The incident is {{ incident_id }}.
Keep this context available.
Keep this context available.
Context stays available.
Context stays available.
Do not remove details.
Do not remove details.
Use neutral language.
Use neutral language.
Escalation follows.
Escalation follows.""",
    )

    report = lint_prompt(prompt)

    assert report.clean is False
    assert prompt.context is not None
    assert report.to_record()["metrics"] == {
        "body_characters": len(prompt.context),
        "body_lines": 12,
        "variables": ["incident_id"],
        "template_parse_error": None,
    }
    assert [issue.code for issue in report.issues] == [
        "LINT001",
        "LINT002",
        "LINT003",
        "LINT004",
        "LINT005",
    ]
    assert all(issue.severity == "warning" for issue in report.issues)


def test_lint_prompt_keeps_template_parse_error_as_non_lint_metric() -> None:
    """Syntax remains a prompt-validate concern, not an advisory lint assertion."""
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Broken template",
        description="Describe inputs and expected output.",
        category="Testing",
        context="Summarize {{ unclosed",
    )

    report = lint_prompt(prompt)

    assert report.variables == ()
    assert report.template_parse_error is not None
    assert "unexpected end" in report.template_parse_error.casefold()
    assert [issue.code for issue in report.issues] == []


def test_lint_prompt_accepts_documented_structured_prompt() -> None:
    """A concise structured prompt with input context should lint cleanly."""
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Review repository",
        description="Review repository input and report actionable risks.",
        category="Engineering",
        context="""# Task
Review {{ repository }} and summarize the highest-priority risks.

# Output
Provide concise findings with evidence.""",
    )

    report = lint_prompt(prompt)

    assert report.clean is True
    assert report.issues == ()
