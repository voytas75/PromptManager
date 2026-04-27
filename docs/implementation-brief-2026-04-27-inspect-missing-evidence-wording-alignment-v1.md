# PromptManager — Implementation Brief

Date: 2026-04-27
Status: proposed
Feature: Inspect Missing-Evidence Wording Alignment v1
Primary sources:
- `docs/product-ssot.md`
- `docs/product-roadmap-ssot.md`
- `docs/canonical-usage-path-v1.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded inspect/detail wording pass so `Recommended next action:` stays action-oriented when run evidence exists but is still too thin for a confident comparable-run recommendation.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse,
- refine.

It should reduce friction between:
- "I can already see that the decision comes from limited run evidence"
- and
- "the next step tells me what to do next instead of restating the evidence gap as a diagnosis"

## Scope

### In scope
- tighten the existing missing-evidence wording returned on the `Recommended next action` seam
- keep `Decision`, `Decision provenance`, and `Last run` evidence cues unchanged in role and data source
- add focused deterministic regression coverage for the bounded wording paths across both detail surfaces
- update the changelog trail for this bounded inspect/detail wording pass

### Out of scope
- new cues or labels
- new evidence sources
- new persistence or schema changes
- broader inspect/detail redesign
- changing decision-selection logic
- changing run-summary or provenance-summary semantics

## Recommended UX posture

Prefer one quiet wording-alignment pass on the existing inspect/detail seams.

Suggested v1:
- keep `Decision` responsible for the high-level recommendation
- keep `Decision based on ...` responsible for explaining the evidence source
- keep `Last run ...` responsible for compact factual run evidence
- make `Recommended next action` consistently read like an operator action

Default recommendation:
- **make the current next step more actionable, not more analytical**

Reason:
- small scope
- directly improves inspect clarity on an existing seam
- matches the product rule to prefer wording/order fixes on current cues before adding new surfaces

## Data source

Use only the existing execution-history evidence already available to inspect/detail logic.

Relevant seams:
- existing recent execution history
- existing comparable-run readiness checks
- existing limited-evidence reason mapping
- existing shared/template detail widget update path

Do not add new persistence in this slice.

## Likely implementation seam

### Wording mapping
- `gui/workspace_history_controller.py`
  - `_build_next_action_summary(...)`
  - `_build_missing_evidence_reason(...)`
  - any small local helper used to map missing-evidence states to action-oriented next steps

### Tests
- `tests/test_workspace_history_controller.py`
- `tests/test_prompt_detail_widget.py` if visible wording expectations need alignment

### Docs
- `docs/CHANGELOG.md`

## Happy-path scenario

1. User inspects a prompt with limited run evidence.
2. `Decision` still reads as the current bounded recommendation.
3. `Decision based on limited run evidence` still explains why confidence is limited.
4. `Last run ...` still shows freshness/readiness facts.
5. `Recommended next action` tells the operator what to do next in action wording.

That is enough for v1.

## Acceptance checks

1. Missing-evidence inspect paths keep using the current bounded seams only.
2. `Decision` stays unchanged for the covered paths.
3. `Decision provenance` stays unchanged for the covered paths.
4. `Run summary` stays unchanged for the covered paths.
5. `Recommended next action` becomes action-oriented for missing-evidence paths.
6. Shared detail and template detail stay aligned.
7. Focused regression coverage protects the bounded wording behavior.

## Suggested wording direction

Use action-oriented wording such as:
- single recent run only -> `Validate before reuse`
- single stale run only -> `Validate before reuse`
- no comparable baseline yet -> `Run another version before comparing`
- missing rating for comparison -> `Add ratings before comparing`
- missing duration for comparison -> `Run again before comparing`

Exact wording may be adjusted during implementation if a nearby existing phrase fits better, but the cue should remain action-first.

## Suggested test

One focused controller cluster is enough.

Recommended shape:
- verify fresh single-run path keeps the same decision/provenance/run-summary cues while next action becomes action-oriented
- verify stale single-run path keeps the safe validation-first next action
- verify no-baseline path maps to an action instead of an `Evidence:` phrase
- verify missing-rating and missing-duration paths map to operator actions instead of diagnostic-only wording
- verify shared and template detail surfaces stay aligned

## Rollback

Rollback should be one isolated patch:
- revert the next-action wording mapping for missing-evidence paths
- revert the focused regression expectations
- revert the changelog entry
- leave the rest of inspect/detail logic untouched

## Anti-goals

- do not add a new evidence panel
- do not redesign inspect/detail layout
- do not change run-summary composition
- do not widen into review workflow or approvals
- do not add new persistence for evaluation state

## Notes for implementation

- Keep the slice boring.
- Action wording beats diagnostic wording on the next-action seam.
- The evidence explanation already has a home; do not duplicate it.