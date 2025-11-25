"""Prompt engineering helpers façade.

Updates: v0.14.1 - 2025-11-25 - Add module history metadata per AGENTS guidelines.

This thin wrapper allows :class:`core.prompt_manager.PromptManager` and other
call-sites to depend on a stable abstraction rather than the concrete
implementation in :pymod:`core.prompt_engineering`. Future iterations can
replace the underlying engine or add pre/post-processing without touching
import sites.
"""

from __future__ import annotations

from typing import Sequence

from ..prompt_engineering import PromptEngineer, PromptEngineeringError, PromptRefinement

__all__ = [
    "PromptEngineerFacade",
    "PromptEngineeringError",
    "PromptRefinement",
]


class PromptEngineerFacade:
    """Facade over :class:`core.prompt_engineering.PromptEngineer`."""

    def __init__(self, model_name: str | None = None):
        self._engineer = PromptEngineer(model_name=model_name) if model_name else PromptEngineer()

    def refine(self, prompt_text: str, *, hints: Sequence[str] | None = None) -> PromptRefinement:  # noqa: D401,E501
        """Return a refined prompt based on provided hints."""

        return self._engineer.refine(prompt_text, hints=hints)

    async def arefine(self, prompt_text: str, *, hints: Sequence[str] | None = None) -> PromptRefinement:  # noqa: D401,E501
        """Async variant of :meth:`refine`."""

        return await self._engineer.arefine(prompt_text, hints=hints)
