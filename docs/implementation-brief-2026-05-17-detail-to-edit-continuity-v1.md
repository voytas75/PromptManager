# PromptManager - Implementation Brief

Date: 2026-05-17
Status: proposed
Feature: Detail-to-edit continuity v1

## Goal

Close one wording gap on the existing inspect/detail seam so a refine-first decision from `WorkspaceHistoryController` hands off to the concrete detail action already supported by `PromptDetailWidget`.

## Problem

Today the controller can emit:
- `Decision: Refine before reuse`
- `Recommended next action: Refine before reuse`

But the shared detail widget only shows the visible handoff cue for the concrete edit wording:
- `Recommended next action: Edit Prompt before reuse`

That leaves the refine-first path internally consistent in the controller, but not continuous on the live detail seam.

## Scope

### In scope
- keep the existing refine-first decision wording
- narrow the next-action mapping for that decision to the existing concrete edit handoff wording
- update focused regression coverage on the controller seam
- rely on the existing detail-widget edit-handoff rendering coverage

### Out of scope
- new decision states
- new widget branches or action buttons
- changes to inspect, validate, baseline, or fork wording outside this refine/edit path
- product or workflow expansion beyond the current inspect/detail seam

## Implementation seam

- `gui/workspace_history_controller.py`
  - update the bounded decision-to-next-action mapping for `Refine before reuse`
- `tests/test_workspace_history_controller.py`
  - lock the new next-action wording for the refine-first lineage path
  - lock the shared decision-to-next-action helper expectation

## Acceptance checks

1. A refine-first decision still renders as `Refine before reuse`.
2. Its next action becomes `Edit Prompt before reuse`.
3. Shared and template detail surfaces stay aligned through the controller seam.
4. Existing detail-widget edit-handoff coverage remains the rendering contract.
5. No unrelated next-action mappings change.

## Validation plan

- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py`
- `.venv/bin/pyright gui/workspace_history_controller.py tests/test_workspace_history_controller.py` if the local run stays cheap

## Rollback

Revert the single controller mapping adjustment, the focused controller expectations, and the changelog note. Leave the shared detail-widget cue logic unchanged.
