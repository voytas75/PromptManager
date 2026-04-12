# PromptManager — Implementation Review

Date: 2026-04-12
Target: `Favorites v1`
Expected sources:
- `docs/implementation-brief-2026-04-12-favorites-v1.md`
- `docs/delegation-brief-2026-04-12-favorites-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change stays inside the intended seam: one prompt-level favorite state, one detail-flow toggle, one favorites-only list filter, and focused regression coverage. It does not widen into collections, pinning, saved sets, or retrieval redesign.

## What matches

### 1. Favorite state is prompt-local and persisted through the existing storage path
The implementation adds one prompt-level `is_favorite` state on `Prompt`, but persists it through the existing `ext5` metadata/storage path instead of adding a new subsystem or broad schema branch.

Touched seams:
- `models/prompt_model.py`
- `core/repository/prompts.py`
- `tests/test_prompt_model_roundtrip.py`
- `tests/test_prompt_manager_storage.py`

That is the smallest reasonable persistence approach for this slice.

### 2. The toggle lives in the shared detail flow
The shared detail widget now exposes exactly one bounded favorite action:
- `Add Favorite`
- `Remove Favorite`

The toggle is wired through the existing prompt update path and used by both detail-widget wiring surfaces.

Touched seams:
- `gui/widgets/prompt_detail_widget.py`
- `gui/main_window_bootstrapper.py`
- `gui/main_window.py`
- `gui/main_view_binder.py`
- `gui/main_window_handlers.py`
- `tests/test_prompt_detail_widget.py`

### 3. Favorites-only retrieval stays inside the existing filter surface
The list/filter change is one added checkbox in the existing filter panel:
- `Favorites only`

Filtering then happens through the existing `PromptListCoordinator.apply_filters` seam.

Touched seams:
- `gui/widgets/prompt_filter_panel.py`
- `gui/prompt_list_coordinator.py`
- `tests/test_prompt_list_coordinator.py`

This matches the brief's requirement for one favorites-only retrieval path without redesigning search or ranking.

### 4. Focused regression coverage matches the shipped behavior
Focused tests now protect:
- prompt favorite serialization/roundtrip
- repository persistence of favorite state
- detail-flow favorite toggle exposure
- favorites-only filtering behavior

Validation passed:
- `pytest -q tests/test_prompt_model_roundtrip.py tests/test_prompt_manager_storage.py tests/test_prompt_detail_widget.py tests/test_prompt_list_coordinator.py`
- result: `33 passed`

## What is missing

Nothing material relative to the brief.

There is intentionally no:
- collections UI
- pinning system
- bulk favorite manager
- saved-set workflow
- recommendation or ranking logic
- broad filter/search redesign

## What drifted / widened

No meaningful scope drift is visible.

The only slightly tempting expansion area was filter-state persistence across sessions. That was intentionally left alone to keep the slice local to prompt storage, detail action, and list filtering, instead of widening into broader layout-preference work.

## Recommended next action

Treat `Favorites v1` as delivered.

If there is a future follow-up, it should be another tiny retrieval slice grounded in real use, not a jump to collections or pinning.

## Sources reviewed

- `docs/implementation-brief-2026-04-12-favorites-v1.md`
- `docs/delegation-brief-2026-04-12-favorites-v1.md`
- `models/prompt_model.py`
- `core/repository/prompts.py`
- `gui/widgets/prompt_detail_widget.py`
- `gui/widgets/prompt_filter_panel.py`
- `gui/prompt_list_coordinator.py`
- focused validation result:
  - `pytest -q tests/test_prompt_model_roundtrip.py tests/test_prompt_manager_storage.py tests/test_prompt_detail_widget.py tests/test_prompt_list_coordinator.py` → `33 passed`
