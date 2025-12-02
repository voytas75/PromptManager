"""State helpers for the Enhanced Prompt Workbench.

Updates:
  v0.1.0 - 2025-11-29 - Introduce workbench session, variable, and execution models.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from core.templating import SchemaValidationMode
from models.category_model import slugify_category
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Mapping, Sequence
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Mapping = _Any
    Sequence = _Any


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class WorkbenchVariable:
    """Tracked placeholder metadata plus sample values for preview rendering."""

    name: str
    description: str | None = None
    sample_value: str | None = None
    last_test_value: str | None = None

    def resolved_value(self) -> str | None:
        """Return the best-known value for previews or execution runs."""
        return self.sample_value or self.last_test_value


@dataclass(slots=True)
class WorkbenchExecutionRecord:
    """Execution metadata captured while iterating on a prompt draft."""

    request_text: str
    response_text: str
    duration_ms: int | None = None
    success: bool = True
    rating: float | None = None
    feedback: str | None = None
    variables: Mapping[str, str] = field(default_factory=dict)
    suggested_focus: str | None = None
    created_at: datetime = field(default_factory=_now)


@dataclass(slots=True)
class WorkbenchSession:
    """Mutable prompt state shared by the Workbench wizard, editor, and preview."""

    prompt_name: str = ""
    goal_statement: str = ""
    audience: str = ""
    system_role: str = ""
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    variables: dict[str, WorkbenchVariable] = field(default_factory=dict)
    template_text: str = ""
    manual_override: bool = False
    schema_text: str = ""
    schema_mode: SchemaValidationMode = SchemaValidationMode.NONE
    execution_history: list[WorkbenchExecutionRecord] = field(default_factory=list)

    def update_from_wizard(
        self,
        *,
        prompt_name: str | None = None,
        goal: str | None = None,
        system_role: str | None = None,
        context: str | None = None,
        audience: str | None = None,
        constraints: Sequence[str] | None = None,
        variables: Mapping[str, WorkbenchVariable] | None = None,
    ) -> str:
        """Merge wizard inputs and rebuild the composite template text."""
        if prompt_name is not None:
            self.prompt_name = prompt_name.strip()
        if goal is not None:
            self.goal_statement = goal.strip()
        if system_role is not None:
            self.system_role = system_role.strip()
        if context is not None:
            self.context = context.strip()
        if audience is not None:
            self.audience = audience.strip()
        if constraints is not None:
            cleaned = [item.strip() for item in constraints if item and item.strip()]
            self.constraints = cleaned
        if variables:
            for name, variable in variables.items():
                self.variables[name] = variable
        composed = self.compose_template()
        self.template_text = composed
        self.manual_override = False
        return composed

    def compose_template(self) -> str:
        """Return a markdown/Jinja template built from the current sections."""
        sections: list[str] = []
        if self.system_role.strip():
            sections.append("### System Role\n" + self.system_role.strip())
        if self.context.strip():
            sections.append("### Context\n" + self.context.strip())
        if self.constraints:
            constraint_lines = [f"- {item}" for item in self.constraints]
            sections.append("### Constraints\n" + "\n".join(constraint_lines))
        if self.variables:
            variable_lines: list[str] = []
            for name in sorted(self.variables):
                variable = self.variables[name]
                line = f"- `{name}` => {{{{ {name} }}}}"
                if variable.description:
                    line += f" — {variable.description.strip()}"
                variable_lines.append(line)
            sections.append("### Variables\n" + "\n".join(variable_lines))
        if self.goal_statement.strip():
            sections.append("### Goal\n" + self.goal_statement.strip())
        return "\n\n".join(section for section in sections if section).strip()

    def set_template_text(self, text: str, *, source: str = "editor") -> None:
        """Persist manual edits so the preview and exports stay in sync."""
        self.template_text = text
        self.manual_override = source == "editor"

    def add_constraint(self, text: str) -> None:
        """Append a new constraint if it is not already present."""
        cleaned = text.strip()
        if not cleaned:
            return
        if cleaned in self.constraints:
            return
        self.constraints.append(cleaned)

    def link_variable(
        self,
        name: str,
        *,
        sample_value: str | None = None,
        description: str | None = None,
    ) -> WorkbenchVariable:
        """Create or update a variable entry used by the preview pane."""
        key = name.strip()
        if not key:
            raise ValueError("Variable name cannot be empty.")
        variable = self.variables.get(key, WorkbenchVariable(name=key))
        if description:
            variable.description = description.strip()
        if sample_value is not None:
            stripped = sample_value.strip()
            variable.sample_value = stripped or None
        self.variables[key] = variable
        return variable

    def variable_payload(self) -> dict[str, str]:
        """Return the subset of variables with concrete values."""
        payload: dict[str, str] = {}
        for name, variable in self.variables.items():
            value = variable.sample_value or variable.last_test_value
            if value:
                payload[name] = value
        return payload

    def record_execution(self, record: WorkbenchExecutionRecord) -> None:
        """Append ``record`` to the in-memory execution history."""
        self.execution_history.append(record)

    def clear_history(self) -> None:
        """Forget recorded executions after exporting a prompt."""
        self.execution_history.clear()

    def suggest_refinement_target(self, response_text: str) -> str:
        """Heuristically pick the editor section most likely needing updates."""
        lower_constraints = " ".join(self.constraints).lower()
        if "json" in lower_constraints and not response_text.strip().startswith("{"):
            return "output"
        if "length" in lower_constraints or "word" in lower_constraints:
            words = len(response_text.split())
            limit = self._word_limit_from_constraints()
            if limit is not None and words > limit:
                return "constraints"
            if limit is None and words > 200:
                return "constraints"
        if "tone" in lower_constraints:
            return "system"
        return "context"

    def update_schema(self, text: str, *, mode: SchemaValidationMode | str | None) -> None:
        """Persist schema settings so exports capture validation rules."""
        self.schema_text = text
        if isinstance(mode, SchemaValidationMode):
            self.schema_mode = mode
        elif isinstance(mode, str):
            self.schema_mode = SchemaValidationMode.from_string(mode)

    def build_prompt(
        self,
        *,
        category: str | None = None,
        language: str = "en",
        tags: Sequence[str] | None = None,
        author: str | None = None,
        prompt_id: uuid.UUID | None = None,
    ) -> Prompt:
        """Hydrate a ``Prompt`` instance from the current draft."""
        category_value = (category or "Workbench").strip() or "Workbench"
        description = self.goal_statement or self.context or "Workbench draft"
        name = self.prompt_name or description or "Workbench Draft"
        prompt = Prompt(
            id=prompt_id or uuid.uuid4(),
            name=name,
            description=description,
            category=category_value,
            category_slug=slugify_category(category_value),
            tags=[tag.strip() for tag in (tags or []) if tag and tag.strip()],
            language=language or "en",
            context=self.template_text.strip(),
            scenarios=[self.goal_statement] if self.goal_statement else [],
            example_input=self.examples[0] if self.examples else None,
            example_output=self.examples[1] if len(self.examples) > 1 else None,
            author=author,
        )
        return prompt

    def _word_limit_from_constraints(self) -> int | None:
        for constraint in self.constraints:
            match = re.search(r"(\d+)", constraint)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None
