# PromptManager — Implementation Brief

Date: 2026-04-27
Status: proposed
Feature: Inspect Decision-Provenance Wording Polish v1
Primary sources:
- `docs/product-ssot.md`
- `docs/product-roadmap-ssot.md`
- `docs/canonical-usage-path-v1.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded inspect/detail wording pass so the provenance cue under `Decision` reads more cleanly and consistently across run-based, limited-evidence, and lineage-only states.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse,
- refine.

It should reduce friction between:
- "the detail view already tells me what the decision is"
- and
- "the provenance cue explains where that decision comes from without sounding heavier or more technical than the rest of inspect"

## Scope

### In scope
- tighten the wording of the existing decision-provenance cue on the current inspect/detail seam
- keep the provenance cue present only where it already appears today
- keep the same provenance categories and branching behavior
- add focused deterministic regression coverage for the adjusted wording across shared and template detail surfaces
- update the changelog trail for this bounded wording pass

### Out of scope
- new provenance states or evidence sources
- new labels or new inspect/detail surfaces
- changes to decision-selection logic
- changes to next-action mapping
- changes to run-summary composition
- broader inspect/detail redesign

## Recommended UX posture

Prefer one quiet wording polish pass on the existing provenance seam.

Suggested v1:
- keep the cue directly under `Decision`
- keep it shorter and more operator-facing
- avoid making the cue sound like a report header or analysis mode
- preserve the current semantic split between:
  - comparable-run evidence,
  - limited run evidence,
  - fork-lineage-only evidence

Default recommendation:
- **make provenance easier to scan, not richer**

Reason:
- small scope
- directly improves inspect readability on an existing seam
- keeps the product aligned with compact decision-support cues rather than analytical expansion

## Data source

Use only the existing evidence already available to the current decision-provenance logic.

Relevant seams:
- existing comparable-run recommendation detection
- existing limited-evidence detection
- existing fork-lineage detection
- existing shared/template detail widget update path

Do not add new persistence in this slice.

## Likely implementation seam

### Wording mapping
- `gui/workspace_history_controller.py`
  - `_build_decision_provenance_summary(...)`
  - any small local helper used to keep provenance wording compact and consistent

### Tests
- `tests/test_workspace_history_controller.py`
- `tests/test_prompt_detail_widget.py` if visible wording expectations need alignment

### Docs
- `docs/CHANGELOG.md`

## Happy-path scenario

1. User inspects a prompt.
2. `Decision` still shows the bounded recommendation.
3. The provenance cue below it still appears only when helpful.
4. The cue reads quickly and clearly as the source of the current recommendation.
5. The operator does not have to parse long phrasing to understand whether the recommendation comes from runs, limited evidence, or lineage only.

That is enough for v1.

## Acceptance checks

1. Decision-provenance cue stays on the existing seam only.
2. Comparable-run provenance still maps to the same underlying state.
3. Limited-evidence provenance still maps to the same underlying state.
4. Fork-lineage-only provenance still maps to the same underlying state.
5. Wording becomes shorter and more consistent.
6. Shared detail and template detail stay aligned.
7. Focused regression coverage protects the bounded wording behavior.

## Suggested wording direction

Use a compact pattern such as:
- `Based on latest 2 comparable runs`
- `Based on limited run evidence`
- `Based on fork lineage only`

Exact wording may be adjusted during implementation if a nearby existing phrase reads better, but the cue should remain short and clearly subordinate to `Decision`.

## Suggested test

One focused controller cluster is enough.

Recommended shape:
- verify comparable-run provenance uses the shortened wording on both detail surfaces
- verify limited-evidence provenance uses the shortened wording on both detail surfaces
- verify lineage-only provenance uses the shortened wording on both detail surfaces
- verify no-provenance paths stay unchanged

## Rollback

Rollback should be one isolated patch:
- revert the provenance wording mapping
- revert the focused regression expectations
- revert the changelog entry
- leave the rest of inspect/detail logic untouched

## Anti-goals

- do not add a provenance panel
- do not add new evidence states
- do not redesign the detail widget
- do not widen into compare/review workflow
- do not add new persistence for explanation state

## Notes for implementation

- Keep the slice boring.
- Provenance should read as a quiet source cue, not a second recommendation.
- Prefer shorter wording if meaning stays obvious.
