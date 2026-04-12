# PromptManager — Delegation Brief

Date: 2026-04-12
Status: ready-for-delegation
Target: dev / Codex
Feature: Template-Aware Workspace CTA v1
Primary brief:
- `docs/implementation-brief-2026-04-12-template-aware-workspace-cta-v1.md`

## Mission

Implement one **small template-aware handoff slice** in the shared prompt detail flow.

Goal:
Make it more obvious that `Open in Workspace` is the correct next reuse action for prompts with detected template variables, **without adding a second primary button**.

This is **not** a new template workflow.
It is a bounded reuse-clarity pass.

## Required posture

Keep this slice:
- small,
- boring,
- local in effect,
- implementation-minded,
- free of adjacent cleanup.

If the UI starts to feel like it has two competing actions, stop and simplify.

## Source anchors

Read first:
- `docs/implementation-brief-2026-04-12-template-aware-workspace-cta-v1.md`
- `docs/implementation-brief-2026-04-11-template-variable-cue-v1.md`
- `docs/implementation-brief-2026-04-11-template-workspace-handoff-cue-v1.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

Likely implementation seam:
- `gui/widgets/prompt_detail_widget.py`

Likely tests to extend:
- `tests/test_prompt_detail_widget.py`

## Deliverable

Ship exactly one bounded patch that does all of the following:
1. strengthens the existing `Open in Workspace` handoff for prompts with detected variables
2. adds one visible but subtle clarification near the existing reuse area
3. keeps the current single-action mental model
4. adds focused regression coverage

## Do now

### 1. Use the existing handoff
Keep `Open in Workspace` as the main action.
Do not add a new primary button such as `Use Template` or `Work with Template`.

### 2. Add one bounded visible clarifier
For prompts with detected variables only, add one small visible clarification near the existing reuse area.

Acceptable shapes:
- a short helper line near the reuse actions
- a bounded secondary caption/state cue attached to the existing workspace action
- a modest template-aware visible hint that makes the current handoff more legible

Not acceptable:
- a second main CTA
- a new panel
- a variable editor in the detail view
- a template wizard

### 3. Preserve existing behavior
Keep unchanged unless strictly required:
- `Copy Prompt`
- `Open in Workspace`
- their action semantics
- plain-prompt behavior

### 4. Add focused tests
Cover at least:
- visible template-aware CTA clarification for template prompts
- no extra cue for plain prompts
- unchanged action labels and enabled-state behavior

## Acceptance checks

1. A prompt with detected template variables shows one visible, bounded clarification that `Open in Workspace` is the right next handoff.
2. `Open in Workspace` remains the single main action path for this case.
3. No second competing primary CTA is introduced.
4. Plain prompts do not show the template-specific visible cue.
5. Existing action semantics remain unchanged.
6. Focused regression coverage protects template-aware and plain-prompt states.
7. No new panel, wizard, inline variable editor, or broader workflow is introduced.

## Validation

Run focused validation only.
Prefer the narrowest reasonable test set proving the slice:
- `tests/test_prompt_detail_widget.py`
- plus any adjacent focused test only if touched

## Required final report

Return:
1. what changed,
2. exact files changed,
3. validation run and results,
4. whether the slice stayed bounded,
5. whether there was any temptation toward a second CTA and how it was avoided.

## Anti-goals

- do not add a second button
- do not create a parallel template flow
- do not redesign the detail reuse area broadly
- do not change workspace destination semantics
- do not touch template preview, search, capture, or fork seams in this slice
- do not bundle another product slice into the same patch

## Rollback

Rollback should be one isolated revert of:
- the visible template-aware CTA clarification,
- any tiny helper added only for this slice,
- the focused regression tests.
