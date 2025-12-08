"""Unit tests for the Enhanced Prompt Workbench session helpers.

Updates:
  v0.1.0 - 2025-11-29 - Cover wizard updates, variable handling, and prompt exports.
"""

from __future__ import annotations

from gui.workbench.session import WorkbenchSession


def test_update_from_wizard_populates_sections() -> None:
    """Populate session template fields when wizard data is provided."""
    session = WorkbenchSession()
    session.update_from_wizard(
        prompt_name="Draft",
        goal="Summarise articles",
        system_role="You are a helpful assistant.",
        context="Summarise the provided article in two sentences.",
        constraints=["Limit to 80 words."],
    )

    assert "### System Role" in session.template_text
    assert "### Constraints" in session.template_text
    assert session.prompt_name == "Draft"
    assert session.manual_override is False


def test_variable_payload_prefers_sample_value() -> None:
    """Ensure variable payload uses sample values when available."""
    session = WorkbenchSession()
    variable = session.link_variable("topic", sample_value="LLMs", description="Subject")
    variable.last_test_value = "fallback"

    payload = session.variable_payload()

    assert payload["topic"] == "LLMs"


def test_build_prompt_uses_goal_as_description() -> None:
    """Copy the goal statement to prompt description when building prompts."""
    session = WorkbenchSession()
    session.prompt_name = "Workbench Draft"
    session.goal_statement = "Generate FAQs"
    session.template_text = "### System Role\nYou are helpful."

    prompt = session.build_prompt(category="Support", language="en", tags=["faq"])

    assert prompt.description == "Generate FAQs"
    assert prompt.category == "Support"
    assert prompt.tags == ["faq"]
    assert prompt.context is not None and prompt.context.startswith("### System Role")


def test_suggest_refinement_target_flags_constraints() -> None:
    """Suggest 'constraints' when responses exceed configured limits."""
    session = WorkbenchSession()
    session.constraints = ["Limit to 10 words"]
    response = "This answer is definitely longer than we expected " * 5

    assert session.suggest_refinement_target(response) == "constraints"
