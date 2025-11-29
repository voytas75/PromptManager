"""Prompt engineering helpers façade.

Updates:
  v0.14.3 - 2025-11-29 - Gate typing imports and wrap long initializer signature.
  v0.14.2 - 2025-11-27 - Align façade signature with PromptEngineer implementation.
  v0.14.1 - 2025-11-25 - Add module history metadata per AGENTS guidelines.

This thin wrapper allows :class:`core.prompt_manager.PromptManager` and other
call-sites to depend on a stable abstraction rather than the concrete
implementation in :pymod:`core.prompt_engineering`. Future iterations can
replace the underlying engine or add pre/post-processing without touching
import sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..prompt_engineering import PromptEngineer, PromptEngineeringError, PromptRefinement

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "PromptEngineerFacade",
    "PromptEngineeringError",
    "PromptRefinement",
]


class PromptEngineerFacade:
    """Facade over :class:`core.prompt_engineering.PromptEngineer`."""

    def __init__(
        self,
        *,
        engineer: PromptEngineer | None = None,
        model_name: str | None = None,
    ) -> None:
        if engineer is not None:
            self._engineer = engineer
        elif model_name is not None:
            self._engineer = PromptEngineer(model=model_name)
        else:
            raise ValueError("Either an engineer instance or model_name must be provided.")

    def refine(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        negative_constraints: Sequence[str] | None = None,
        structure_only: bool = False,
    ) -> PromptRefinement:
        """Refine a prompt using the underlying engineer."""

        return self._engineer.refine(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
            negative_constraints=negative_constraints,
            structure_only=structure_only,
        )

    def refine_structure(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        negative_constraints: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Run structure-only refinement via :meth:`PromptEngineer.refine_structure`."""

        return self._engineer.refine_structure(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
            negative_constraints=negative_constraints,
        )
