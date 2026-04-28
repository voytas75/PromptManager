# PromptManager — Implementation Brief

Date: 2026-04-28
Status: ready-for-implementation
Feature: Tag Filter Visibility v1
Primary source: bounded follow-up on the existing prompt-library filter panel seam

## Goal

Implement one bounded **Tag Filter Visibility v1** improvement so the existing tag filter makes the active tag choice more visible without redesigning the filter system.

## Why this slice now

PromptManager is asset-first. The main job is to capture, organize, retrieve, inspect, and reuse prompt assets.

The tag filter already exists in the shared prompt-library filter panel and is directly tied to the core retrieval loop, but the current visible state is still sparse:
- the combo defaults to `All tags`
- the active tag context is visible only inside the combo itself
- there is no compact always-visible cue showing that retrieval is currently narrowed by one tag

That makes this a good bounded usability slice on an existing function that is close to the app’s primary purpose.

## Scope

In scope:
- existing `PromptFilterPanel` only
- one compact visible cue for the active tag state
- reuse existing tag selection data only
- no new persistence
- no coordinator/presenter retrieval redesign

Out of scope:
- multi-tag selection
- tag counts
- search ranking changes
- filter-panel redesign
- favorites/category/quality behavior changes beyond necessary coexistence

## Proposed UX

When no tag is selected:
- compact helper text remains calm, e.g. `Tag filter: all tags`

When a tag is selected:
- helper text becomes explicit, e.g. `Tag filter: outages`

The cue should live in the existing filter panel and update whenever the selected tag changes.

## Expected files

- `gui/widgets/prompt_filter_panel.py`
- `tests/test_prompt_filter_panel.py` or nearest focused test file
- `docs/CHANGELOG.md`

## Acceptance checks

1. The filter panel shows a visible compact tag-state cue.
2. Default state reads as all-tags rather than blank/implicit state.
3. Selecting a tag updates the cue immediately.
4. Existing tag selection semantics stay unchanged.
5. The slice remains local to the filter-panel seam.

## Verification plan

- targeted filter-panel tests
- nearby prompt-list smoke if needed
- focused Ruff on touched files

## Notes

Keep this slice wording-first and seam-local. Do not widen it into retrieval-state counts, badges, or filter persistence changes.