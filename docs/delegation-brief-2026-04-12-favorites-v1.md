# PromptManager — Delegation Brief

Date: 2026-04-12
Status: ready-for-delegation
Target: dev / Codex
Feature: Favorites v1
Primary brief:
- `docs/implementation-brief-2026-04-12-favorites-v1.md`

## Mission

Implement one **small Favorites v1** slice.

Goal:
- let the operator mark a prompt as favorite
- let the operator retrieve favorite prompts quickly later
- keep the feature small and near-core

This is **not** a collections or saved-sets feature.
It is a bounded retrieval/reuse support pass.

## Required posture

Keep this slice:
- small,
- boring,
- local in effect,
- implementation-minded,
- free of adjacent cleanup.

If the work starts to look like collections, folders, pinning systems, or recommendation logic, stop and simplify.

## Source anchors

Read first:
- `docs/implementation-brief-2026-04-12-favorites-v1.md`
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

Likely implementation seams:
- `models/prompt_model.py`
- `core/repository/prompts.py`
- `gui/widgets/prompt_detail_widget.py`
- `gui/widgets/prompt_filter_panel.py`
- the smallest presenter/controller seam needed to apply the new filter

Likely tests to extend:
- repository/model tests around prompt persistence
- `tests/test_prompt_detail_widget.py`
- filter/list tests where current prompt filtering behavior is already covered

## Deliverable

Ship exactly one bounded patch that does all of the following:
1. adds one prompt-level favorite state
2. lets the operator toggle that state from the detail flow
3. adds one favorites-only retrieval/filter path
4. persists the state through the current prompt storage path
5. adds focused regression coverage

## Do now

### 1. Add one favorite state
Implement the smallest durable prompt-level favorite flag.
Prefer the current prompt persistence path over broader new storage design.

### 2. Add one detail-flow toggle
Expose one favorite toggle in the existing detail surface.
Keep the interaction simple and local.

### 3. Add one favorites-only filter path
Add one simple way to filter the prompt list to favorites only.
Do not redesign the whole filter surface.

### 4. Add focused tests
Cover at least:
- persisted favorite state
- favorite toggle behavior in the detail surface
- favorites-only filtering behavior

## Acceptance checks

1. A prompt can be marked and unmarked as favorite from the detail flow.
2. Favorite state persists through the existing prompt storage path.
3. The prompt list/filter surface can show favorites only.
4. Non-favorite prompts are excluded when the favorites-only filter is active.
5. No collections, folders, saved sets, recommendation logic, or management panel is introduced.
6. Focused regression coverage protects stored state, toggle behavior, and favorites-only filtering.
7. The slice stays local and does not redesign retrieval broadly.

## Validation

Run focused validation only.
Prefer the narrowest reasonable test set proving the slice.

## Required final report

Return:
1. what changed
2. exact files changed
3. validation run and results
4. whether the slice stayed bounded
5. whether there was any temptation toward collections/pinning scope and how it was avoided

## Anti-goals

- do not widen into collections
- do not add saved sets or folders
- do not add ranking/recommendation logic based on favorites
- do not redesign the whole filter panel
- do not add bulk favorite management
- do not touch unrelated workspace/search/template/fork seams beyond what is strictly needed for the filter path

## Rollback

Rollback should be one isolated revert of:
- favorite-state handling
- the detail toggle
- the favorites-only filter path
- the focused regression tests
