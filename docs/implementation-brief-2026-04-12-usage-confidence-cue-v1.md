# PromptManager — Implementation Brief

Date: 2026-04-12
Status: delivered and spot-verified
Feature: Usage Confidence Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/canonical-usage-path-v1.md`

Validation:
- `pytest -q tests/test_prompt_detail_widget.py`
- result: `18 passed`

## Goal

Implement one bounded **usage confidence cue** inside the shared prompt detail view so an operator can tell whether a prompt is only stored or has already seen some real use.

The desired experience is simple:

> open prompt detail → immediately see whether this asset has already been used in practice

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse,
- refine.

The catalog already stores `usage_count`.
This slice should turn that existing signal into one quiet operator-facing cue at the exact moment when the user is deciding whether a prompt feels trustworthy enough to reuse.

This should stay a **confidence cue**, not a usage analytics feature.

## Scope

### In scope
- show one compact usage cue in the shared detail view when `usage_count > 0`
- derive the cue only from the existing stored `usage_count`
- keep the cue quiet, short, and subordinate to the existing inspect/reuse flow
- add focused regression coverage for visible and absent states

### Out of scope
- last-used timestamps
- usage trend or sparkline UI
- usage-based ranking or sorting
- list-row usage badges
- top-prompts or favorites logic
- analytics/dashboard expansion
- new persistence or schema changes
- any change to execution flow semantics

## Recommended UX posture

Prefer one short confidence signal over richer statistics.

Suggested v1 wording:
- `Reuse signal: used 1 time`
- `Reuse signal: used 4 times`

Recommended rules:
- show nothing when `usage_count <= 0`
- use singular/plural wording correctly
- keep the cue near the existing detail inspection text rather than as a new panel or metric row
- do not imply quality or approval, only prior use

Default recommendation:
- **quiet confidence, not gamified usage**

## Likely implementation seam

### Shared detail widget
- `gui/widgets/prompt_detail_widget.py`
  - extend the existing bounded detail/inspection cue rendering path
  - reuse current prompt data already available to the widget
  - keep layout changes minimal and local

### Tests
- likely seam:
  - `tests/test_prompt_detail_widget.py`
- cover at least:
  - cue visible when `usage_count > 0`
  - singular wording for `1`
  - plural wording for values `> 1`
  - cue absent when `usage_count == 0`

## Happy-path scenarios

### Scenario A: previously used prompt
1. User opens the detail view for a prompt already used in practice.
2. The detail surface shows one quiet usage confidence cue.
3. The user can judge reuse confidence a little faster without opening history or analytics.

### Scenario B: never-used prompt
1. User opens the detail view for a prompt that is only stored.
2. No usage cue is shown.
3. The detail view stays clean and does not add noise for zero-signal cases.

That is enough for v1.

## Acceptance checks

1. A prompt with `usage_count > 0` shows one compact usage confidence cue in the shared detail view.
2. A prompt with `usage_count == 0` does not show the cue.
3. Singular/plural wording is correct.
4. No new panel, dashboard, ranking, or analytics behavior is introduced.
5. Focused regression tests pass.

## Rollback

Rollback should be one isolated patch:
- remove the usage confidence cue from the detail view
- remove the focused regression tests
- leave inspect/reuse/execution flows untouched

## Anti-goals

- do not turn this into usage analytics
- do not add list-row badges or search-ranking changes
- do not infer quality from usage count
- do not widen into favorites, recents logic, or recommendations
- do not redesign the shared detail widget around metrics

## Notes for implementation

- Keep the slice boring.
- Reuse the existing `usage_count` only.
- The cue should reduce hesitation, not add analysis overhead.
- If the UI starts asking for trend/history controls, the slice is drifting.
