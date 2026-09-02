# PromptManager — Implementation Brief

Date: 2026-04-27
Status: proposed
Feature: Inspect No-History Fallback Clarity v1
Primary sources:
- `docs/product-ssot.md`
- `docs/canonical-usage-path-v1.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded inspect/detail clarity pass for prompts with no recorded run history so the inspect path stays calm and legible before any validation evidence exists.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse,
- refine.

It should reduce friction between:
- "I just captured or reopened a prompt that has never been run"
- and
- "the detail view still gives me a clear next step without implying hidden history or analytical certainty"

## Scope

### In scope
- tighten the no-history inspect fallback on the current decision-support seams
- keep the no-history state calm and explicit without adding analytical weight
- add focused deterministic regression coverage for no-history inspect behavior across shared and template detail surfaces
- update the changelog trail for this bounded no-history clarity pass

### Out of scope
- new history panels or empty-state dashboards
- new persistence or schema changes
- analytics or run-scheduling features
- broader inspect/detail redesign
- changes to comparable-run logic
- changes to existing history-backed wording beyond what is required for no-history clarity

## Recommended UX posture

Prefer one quiet fallback pass on the current inspect/detail seams.

Suggested v1:
- prompts with no recorded runs should still feel reusable and editable
- the UI should not imply missing hidden data or background failure
- the fallback should be easy to scan in the same tone as the existing inspect cues
- if a cue is added, it should be one bounded low-noise cue only

Default recommendation:
- **make the no-history state clearer, not heavier**

Reason:
- small scope
- directly supports the prompt-asset loop before prompt execution exists
- improves inspect clarity for a common early-life prompt state

## Data source

Use only the existing prompt and execution-history state already available to the inspect/detail controller.

Relevant seams:
- existing execution-history lookup
- existing decision / next-action summary path
- existing shared/template detail widget update path

Do not add new persistence in this slice.

## Likely implementation seam

### No-history fallback behavior
- `gui/workspace_history_controller.py`
  - `_build_decision_summary(...)`
  - `_build_next_action_summary(...)`
  - any small local helper needed to keep no-history wording bounded and explicit

### Tests
- `tests/test_workspace_history_controller.py`
- `tests/test_prompt_detail_widget.py` only if visible label expectations change

### Docs
- `docs/CHANGELOG.md`

## Happy-path scenario

1. User opens a prompt that has no recorded run history.
2. Inspect/detail does not show misleading run-based confidence.
3. The visible recommendation remains calm and actionable.
4. The operator understands what to do next without thinking the product is hiding missing run data.

That is enough for v1.

## Acceptance checks

1. No-history prompts stay on the existing inspect/detail seams only.
2. No-history prompts do not surface run-based provenance or comparison wording.
3. The fallback wording stays calm and operator-facing.
4. Shared detail and template detail stay aligned.
5. Focused regression coverage protects the bounded no-history behavior.

## Suggested wording direction

One acceptable direction would be to keep `Decision` simple and make the next step explicit for no-history prompts, for example toward:
- `Decision: Reuse as-is`
- `Recommended next action: Validate before reuse`

Exact wording may be adjusted during implementation if the existing seam suggests a cleaner variant, but the result should stay bounded and non-analytical.

## Suggested test

One focused controller cluster is enough.

Recommended shape:
- verify no-history prompt keeps no run-summary cue
- verify no-history prompt keeps no provenance cue
- verify no-history next step is explicit and operator-facing if adjusted
- verify shared and template detail surfaces stay aligned

## Rollback

Rollback should be one isolated patch:
- revert the no-history fallback wording adjustment
- revert the focused regression expectations
- revert the changelog entry
- leave the history-backed inspect/detail logic untouched

## Anti-goals

- do not add a no-history dashboard
- do not add onboarding flow or wizard
- do not add synthetic run evidence
- do not widen into analytics or scheduling
- do not redesign the detail widget

## Notes for implementation

- Keep the slice boring.
- No-history should feel explicit, not empty or alarming.
- Prefer one calm next step over extra explanation text.
