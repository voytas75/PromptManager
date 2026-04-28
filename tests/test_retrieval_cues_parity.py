"""Parity decision tests for retrieval/discovery prompt-list cues.

Updates:
  v0.1.0 - 2026-04-27 - Lock retrieval cues as GUI-local by design
  unless promoted into shared analytics.
"""

from __future__ import annotations

from gui.prompt_list_coordinator import PromptLoadResult
from gui.prompt_list_model import PromptListModel


def test_retrieval_prompt_list_cues_remain_gui_local_by_design() -> None:
    """Prompt-list retrieval cues should not be treated as shared/headless analytics fields."""
    assert hasattr(PromptListModel, "MatchReasonRole")
    result = PromptLoadResult(
        all_prompts=[],
        search_results=None,
        preserve_search_order=False,
        search_error=None,
        operator_state_label="Browsing all prompts",
    )

    assert result.operator_state_label == "Browsing all prompts"
    assert not hasattr(result, "decision_summary")
    assert not hasattr(result, "next_action_summary")
    assert not hasattr(result, "freshness_summary")
