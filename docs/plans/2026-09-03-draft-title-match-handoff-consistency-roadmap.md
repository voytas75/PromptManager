# PromptManager — Draft Title-Match Handoff Consistency

**Status:** verified delivery record
**Date:** 2026-09-03
**Product priority:** Priority 1 — retrieval, inspect, and reuse confidence
**Canonical product SSOT:** `docs/product-ssot.md`

## Purpose

Record the bounded correction that prevents a captured draft from being presented in the prompt list as ready for direct reuse merely because the active search matches its title.

## Confirmed baseline

- `PromptListModel` keeps `Matched in title` as the retrieval reason for title matches.
- Before this slice, that same branch returned `Ready to reuse` even when `prompt.ext2["capture_state"] == "draft"`.
- Detail remains the canonical draft decision surface and already exposes `Promote Draft`.

## Delivered contract

For an active title match:

- a non-draft prompt keeps the existing `Ready to reuse` list handoff;
- a captured draft keeps `Matched in title` but has no list-side direct-reuse handoff;
- the correction does not add a competing list CTA; promotion remains a detail-side action.

## Scope

- `gui/prompt_list_model.py`
- `tests/test_prompt_list_model.py`

Out of scope:

- retrieval ranking or semantic-search behavior;
- new persistence, panels, or workflow state;
- changes to source/scenario/description match handoffs;
- prompt-chain work.

## TDD evidence

- RED in disposable worktree: a draft title-match expected no `HandoffCueRole`, but received `Ready to reuse`.
- GREEN: the list model suppresses that direct-reuse cue only for `capture_state == "draft"`.

## Verification

- `pytest tests/test_prompt_list_model.py -q` → `23 passed`
- `pytest tests/test_prompt_list_model.py tests/test_prompt_detail_widget.py tests/test_prompt_editor_flow.py tests/test_canonical_operator_path_parity.py -q` → `73 passed`
- `pyright gui/prompt_list_model.py tests/test_prompt_list_model.py` → `0 errors`
- Ruff check and formatter check for the two paths → passed.
- Final approved code scope: `21 additions + 1 deletion = 22 gross lines`.

## Closure

This closes the draft/title-match contradiction on the existing list/detail seam. Do not open another wording-only retrieval slice unless a new operator-decision contradiction is reproduced.

**Next selection rule:** choose the next asset-loop seam explicitly from a fresh read-only probe; candidate areas are capture-quality consistency and CLI/GUI retrieval-contract parity.
