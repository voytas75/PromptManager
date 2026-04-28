# PromptManager — Implementation Brief

Date: 2026-04-28
Status: ready-for-implementation
Feature: Execute as Context Empty-State Clarity v1
Primary source: bounded follow-up on the existing `Execute as Context…` action seam

## Goal

Implement one bounded **Execute as Context Empty-State Clarity v1** improvement so the existing `Execute as Context…` context-menu action gives a clearer reason when it is disabled because the selected prompt has no stored prompt body.

## Why this slice now

PromptManager is asset-first, but the core loop includes reuse and refinement through existing actions.

`Execute as Context…` already exists as a reuse-oriented action in the prompt-actions seam. Today it is disabled when a prompt has no stored body (`prompt.context`), but the disabled state has no operator-facing explanation in that menu path.

That is technically correct, but not very helpful. A tiny clarity pass here improves reuse legibility without changing execution behavior.

## Scope

In scope:
- existing `PromptActionsController.show_context_menu()` seam only
- disabled-state explanation for `Execute as Context…`
- one bounded tooltip on the disabled menu action when prompt body is missing
- focused regression coverage for the disabled-action path

Out of scope:
- execution engine changes
- prompt model/storage changes
- workspace handoff changes
- dialog redesign
- new actions or recovery flows

## Proposed UX

When a prompt has no stored body:
- keep `Execute as Context…` disabled
- add a tooltip such as:
  - `Execute as Context requires a stored prompt body. Add prompt text before using this action.`

When a prompt has a stored body:
- keep existing enabled behavior unchanged

## Expected files

- `gui/prompt_actions_controller.py`
- `tests/test_prompt_actions_controller.py`
- `docs/CHANGELOG.md`

## Acceptance checks

1. `Execute as Context…` remains disabled when the prompt has no body.
2. The disabled action exposes a clear bounded tooltip explaining why.
3. Prompts with a stored body keep current behavior.
4. The slice stays local to the prompt-actions context-menu seam.

## Verification plan

- targeted prompt-actions test
- nearby prompt-actions smoke
- focused Ruff on touched files

## Notes

Keep this seam-local and wording-only. Do not widen it into execution fallback logic, editor changes, or new recovery UI.
