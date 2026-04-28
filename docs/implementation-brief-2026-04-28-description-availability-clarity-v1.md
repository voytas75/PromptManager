# PromptManager — Implementation Brief

Date: 2026-04-28
Status: ready-for-implementation
Feature: Description Availability Clarity v1
Primary source: bounded follow-up on the existing `Show Description` action seam

## Goal

Implement one bounded **Description Availability Clarity v1** improvement so the existing `Show Description` action gives a more helpful operator-facing fallback when a prompt has no saved description yet.

## Why this slice now

PromptManager is asset-first. Operators need to retrieve a prompt, understand what it is for, and decide whether it fits before reuse.

`Show Description` already exists as a quick-reference action in the prompt actions seam, but its empty-state copy is still minimal:
- title: `No description available`
- body: `The selected prompt does not have a description yet.`

That is accurate, but not very action-guiding. A small wording improvement here stays close to the product core: understanding prompt assets faster.

## Scope

In scope:
- existing `PromptActionsController.show_prompt_description()` seam only
- empty-description fallback wording only
- one bounded next-step hint for the operator
- focused tests for the empty-description path

Out of scope:
- detail-view redesign
- new metadata fields
- generation of descriptions
- search or inspect logic changes
- workspace or execution flow changes

## Proposed UX

When a prompt has no description:
- keep the existing dialog title: `No description available`
- tighten the message to something like:
  - `The selected prompt does not have a description yet. Inspect the prompt body or add a short description for faster reuse.`

When a prompt has a description:
- keep existing behavior unchanged

## Expected files

- `gui/prompt_actions_controller.py`
- `tests/test_prompt_actions_controller.py`
- `docs/CHANGELOG.md`

## Acceptance checks

1. `Show Description` still opens the existing info dialog.
2. Empty-description prompts show a more helpful next-step message.
3. Non-empty description behavior remains unchanged.
4. The slice stays local to the prompt-actions seam.

## Verification plan

- targeted prompt-actions tests
- nearby prompt-actions smoke
- focused Ruff on touched files

## Notes

Keep this wording-only and seam-local. Do not widen it into description synthesis, fallback generation, or inspect-flow redesign.