"""Prompt engineering helpers façade.

Updates: v0.14.2 - 2025-11-27 - Align façade signature with PromptEngineer implementation.
Updates: v0.14.1 - 2025-11-25 - Add module history metadata per AGENTS guidelines.

This thin wrapper allows :class:`core.prompt_manager.PromptManager` and other
call-sites to depend on a stable abstraction rather than the concrete
implementation in :pymod:`core.prompt_engineering`. Future iterations can
replace the underlying engine or add pre/post-processing without touching
import sites.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..prompt_engineering import PromptEngineer, PromptEngineeringError, PromptRefinement

__all__ = [
    "PromptEngineerFacade",
    "PromptEngineeringError",
    "PromptRefinement",
]


class PromptEngineerFacade:
    """Facade over :class:`core.prompt_engineering.PromptEngineer`."""

    def __init__(self, *, engineer: Optional[PromptEngineer] = None, model_name: Optional[str] = None) -> None:
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
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        negative_constraints: Optional[Sequence[str]] = None,
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
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        negative_constraints: Optional[Sequence[str]] = None,
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
