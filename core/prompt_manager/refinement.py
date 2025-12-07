"""Prompt refinement helpers shared across Prompt Manager components.

Updates:
  v0.1.0 - 2025-12-03 - Extract prompt refinement helpers from PromptManager.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from ..exceptions import PromptEngineeringUnavailable
from ..notifications import NotificationLevel
from ..prompt_engineering import PromptEngineer, PromptEngineeringError, PromptRefinement

if TYPE_CHECKING:
    from collections.abc import Sequence

    from . import PromptManager as _PromptManager
else:
    _PromptManager = Any

__all__ = ["PromptRefinementMixin"]


class PromptRefinementMixin:
    """Mixin encapsulating prompt refinement workflows."""

    def _as_prompt_manager(self) -> _PromptManager:
        """Return self casted to PromptManager for shared dependencies."""
        return cast("_PromptManager", self)

    @property
    def prompt_engineer(self) -> PromptEngineer | None:
        """Return the configured prompt engineering helper, if any."""
        return getattr(self, "_prompt_engineer", None)

    @property
    def prompt_structure_engineer(self) -> PromptEngineer | None:
        """Return the configured structure-only prompt engineering helper."""
        engine = getattr(self, "_prompt_structure_engineer", None)
        if engine is not None:
            return engine
        return self.prompt_engineer

    def refine_prompt_text(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        negative_constraints: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Improve a prompt using the configured prompt engineer."""
        if not prompt_text.strip():
            raise PromptEngineeringError("Prompt refinement requires non-empty prompt text.")
        engineer = self.prompt_structure_engineer or self.prompt_engineer
        if engineer is None:
            raise PromptEngineeringUnavailable(
                "Prompt engineering is not configured. Set PROMPT_MANAGER_LITELLM_MODEL "
                "to enable refinement."
            )
        task_id = f"prompt-refine:{uuid.uuid4()}"
        metadata = {
            "prompt_length": len(prompt_text or ""),
            "has_name": bool(name),
            "tag_count": len(tags or []),
        }
        manager = self._as_prompt_manager()
        with manager._notification_center.track_task(  # noqa: SLF001
            title="Prompt refinement",
            task_id=task_id,
            start_message="Analysing prompt via LiteLLM…",
            success_message="Prompt refined.",
            failure_message="Prompt refinement failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                return engineer.refine(
                    prompt_text,
                    name=name,
                    description=description,
                    category=category,
                    tags=tags,
                    negative_constraints=negative_constraints,
                )
            except PromptEngineeringError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise PromptEngineeringError("Prompt refinement failed unexpectedly.") from exc

    def refine_prompt_structure(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Reformat a prompt to improve structure without changing intent."""
        if not prompt_text.strip():
            raise PromptEngineeringError("Prompt refinement requires non-empty prompt text.")
        engineer = self.prompt_structure_engineer or self.prompt_engineer
        if engineer is None:
            raise PromptEngineeringUnavailable(
                "Prompt engineering is not configured. Set PROMPT_MANAGER_LITELLM_MODEL "
                "to enable refinement."
            )
        task_id = f"prompt-structure-refine:{uuid.uuid4()}"
        metadata = {
            "prompt_length": len(prompt_text or ""),
            "mode": "structure",
        }
        manager = self._as_prompt_manager()
        with manager._notification_center.track_task(  # noqa: SLF001
            title="Prompt structure refinement",
            task_id=task_id,
            start_message="Reformatting prompt via LiteLLM…",
            success_message="Prompt structure refined.",
            failure_message="Prompt structure refinement failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                return engineer.refine_structure(
                    prompt_text,
                    name=name,
                    description=description,
                    category=category,
                    tags=tags,
                )
            except PromptEngineeringError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise PromptEngineeringError("Prompt refinement failed unexpectedly.") from exc
