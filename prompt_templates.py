"""Default prompt templates surfaced across Prompt Manager workflows.

Updates: v0.1.1 - 2025-11-24 - Add category suggestion template for LiteLLM workflows.
Updates: v0.1.0 - 2025-11-23 - Centralise prompt template defaults for runtime overrides.
"""

from __future__ import annotations

from typing import Dict

NAME_GENERATION_PROMPT = (
    "You generate concise, descriptive prompt names for a prompt catalogue. "
    "Return a title of at most 5 words. Avoid punctuation except spaces."
)

DESCRIPTION_GENERATION_PROMPT = (
    "You write concise catalogue descriptions for reusable AI prompts. "
    "Summarise the intent, inputs, and expected outcomes in 2 sentences. "
    "Avoid bullet lists and marketing fluff."
)

SCENARIO_GENERATION_PROMPT = (
    "You help product teams document when to reuse AI prompts. "
    "Given the full prompt text, produce concise, action-oriented usage scenarios. "
    "Respond with a JSON array of unique strings. Each string should begin with a verb and "
    "describe a situation where the prompt is helpful. Keep each scenario under 140 characters."
)

PROMPT_ENGINEERING_PROMPT = (
    "You are a meticulous prompt engineering assistant."
    " Analyse and refine prompts using the following ordered rules."
    "\n\n"
    "### Rules for LLM Prompt Query Language\n"
    "1. Clarity and Specificity: define the task, detail level, tone, and explicitly state"
    " what is out of scope.\n"
    "2. Context Completeness: include relevant background, constraints, and summarise"
    " large context within token budgets.\n"
    "3. Output Structure Control: specify the desired output structure, format, and how"
    " to handle unknowns or uncertainty.\n"
    "4. Role/Goal Separation: optionally assign a role that is limited to the task and"
    " restate the goal without implying sentience.\n"
    "5. Deterministic Language: use imperative phrasing and explicit delimiters to"
    " separate instructions from data.\n"
    "6. Anthropomorphism Control: default to non-anthropomorphic framing unless the"
    " use-case demands a persona.\n"
    "7. Prevent Agency Misinterpretation: avoid implying memory, intent, or autonomy"
    " beyond the prompt context.\n"
    "8. Psychological Transparency: request reasoning steps and confidence when"
    " helpful, balancing brevity and clarity.\n"
    "9. Example-Driven Prompting: include examples (and counter-examples when"
    " relevant) to anchor expectations.\n"
    "10. Iterative Refinement: document refinements, versioning, and measurement"
    " cues for future iterations.\n"
    "11. Hybrid Natural-Formal Syntax: combine natural language with structured"
    " schemas (e.g., JSON sections) when precision matters.\n"
    "12. Guardrails and Safety: surface security, privacy, or policy constraints and"
    " reference handling instructions.\n"
    "13. Token and Resource Efficiency: stay concise, highlight required vs optional"
    " context, and note truncation strategies.\n"
    "14. Role-Play for Engagement: only when the use-case benefits from it and make"
    " it explicit that the persona is simulated.\n"
    "15. Conversational Polishing: optional—politeness is secondary to clarity.\n"
    "Always preserve the author's intent while fixing omissions or ambiguities."
    " Return precise, security conscious prompts with actionable structure."
    " Produce deterministic guidance that can run unchanged in production systems."
    "\n\nReturn a JSON object with the keys: analysis (string), improved_prompt (string),"
    " checklist (array of strings), warnings (array of strings), and confidence (number 0-1)."
    " If no changes are warranted set improved_prompt equal to the original but justify"
    " the decision in analysis."
)

CATEGORY_GENERATION_PROMPT = (
    "You classify AI prompts into an existing set of product categories. "
    "Select the single best category label from the provided list. "
    "Return only the label text without explanations or extra words."
)

PROMPT_TEMPLATE_KEYS = (
    "name_generation",
    "description_generation",
    "scenario_generation",
    "prompt_engineering",
    "category_generation",
)

PROMPT_TEMPLATE_LABELS: Dict[str, str] = {
    "name_generation": "Prompt name suggestions",
    "description_generation": "Prompt description synthesis",
    "scenario_generation": "Scenario drafting",
    "prompt_engineering": "Prompt refinement",
    "category_generation": "Prompt category suggestions",
}

PROMPT_TEMPLATE_DESCRIPTIONS: Dict[str, str] = {
    "name_generation": "LLM system prompt used when generating names for new prompts.",
    "description_generation": "LLM system prompt used to auto-summarise prompt descriptions.",
    "scenario_generation": "LLM system prompt used to draft usage scenarios for prompts.",
    "prompt_engineering": "LLM system prompt used when refining prompts with the engineer.",
    "category_generation": "LLM system prompt used to classify prompts into catalogue categories.",
}

DEFAULT_PROMPT_TEMPLATES: Dict[str, str] = {
    "name_generation": NAME_GENERATION_PROMPT,
    "description_generation": DESCRIPTION_GENERATION_PROMPT,
    "scenario_generation": SCENARIO_GENERATION_PROMPT,
    "prompt_engineering": PROMPT_ENGINEERING_PROMPT,
    "category_generation": CATEGORY_GENERATION_PROMPT,
}


def get_default_prompt(key: str) -> str:
    """Return the default prompt text for a workflow key."""

    return DEFAULT_PROMPT_TEMPLATES.get(key, "")


def normalise_prompt_templates(value: Dict[str, str] | None) -> Dict[str, str]:
    """Return a cleaned mapping of prompt overrides keyed by workflow."""

    if not value:
        return {}
    cleaned: Dict[str, str] = {}
    for key, text in value.items():
        if key not in PROMPT_TEMPLATE_KEYS:
            continue
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if stripped:
            cleaned[key] = stripped
    return cleaned


__all__ = [
    "PROMPT_TEMPLATE_KEYS",
    "PROMPT_TEMPLATE_LABELS",
    "PROMPT_TEMPLATE_DESCRIPTIONS",
    "DEFAULT_PROMPT_TEMPLATES",
    "NAME_GENERATION_PROMPT",
    "DESCRIPTION_GENERATION_PROMPT",
    "SCENARIO_GENERATION_PROMPT",
    "PROMPT_ENGINEERING_PROMPT",
    "CATEGORY_GENERATION_PROMPT",
    "get_default_prompt",
    "normalise_prompt_templates",
]
