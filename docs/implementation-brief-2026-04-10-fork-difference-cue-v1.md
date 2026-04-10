# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Fork Difference Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-review-2026-04-10-product-alignment-and-fork-state.md`

## Goal

Implement one bounded **Fork Difference Cue** improvement so a forked prompt can show one compact, human-readable summary of what materially differs from its parent without adding a compare screen, diff workflow, or new lineage surface.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse,
- refine.

It should reduce friction between:
- "I can see this prompt is a fork"
- and
- "I can quickly judge why this fork might be the better prompt to use"

without widening PromptManager into a diff-heavy history product.

## Scope

### In scope
- detect one bounded set of meaningful differences between a fork and its direct parent
- render one compact read-only cue in the existing detail flow, for example:
  - `Changed from parent: body, tags`
- keep the cue quiet and summary-level only
- add focused regression coverage for visible and hidden states

### Out of scope
- full text diff UI
- version-to-version compare flows
- ancestry graphs
- fork rationale capture
- merge or review workflows
- schema changes
- new panels, dialogs, or tabs

## Recommended UX posture

Prefer one small inspection cue over a richer compare experience.

Suggested v1:
- show the cue only for prompts with a resolvable direct parent fork link
- compare only a tight bounded field set:
  - prompt body
  - description
  - tags
  - source
- show only labels for changed fields, not field values
- hide the cue entirely when no bounded difference is worth surfacing

Default recommendation:
- **make fork differences more legible, not more detailed**

Reason:
- small scope
- directly supports prompt selection and reuse decisions
- builds on the new lineage clarity work without opening a new product surface

## Data source

Use only existing prompt records and the existing direct parent fork link.

Relevant seams:
- prompt parent fork lookup
- existing prompt retrieval by id
- current prompt detail rendering path

Do not add new persistence in this slice.

## Likely implementation seam

### Difference summary logic
- `gui/workspace_history_controller.py`
  - extend the current lineage/update path with one bounded parent-difference summary helper
  - resolve the direct parent prompt using the existing fork link
  - compare only the selected bounded fields

### Detail rendering
- reuse the existing prompt detail information area
- either append the cue into the lineage summary or add one small adjacent detail cue if the current widget already supports it cleanly
- prefer the smaller change

### Tests
- `tests/test_workspace_history_controller.py`
- add focused coverage for:
  - visible `Changed from parent: ...` summary
  - hidden state for non-fork prompts or no meaningful bounded differences

## Happy-path scenario

1. User opens a forked prompt.
2. The existing lineage summary already shows `Forked from <prompt name>`.
3. The detail view also shows one compact cue such as `Changed from parent: body, tags`.
4. The user can quickly judge whether this fork is meaningfully different without opening another screen.

That is enough for v1.

## Acceptance checks

1. A forked prompt with bounded field differences shows one compact `Changed from parent: ...` cue.
2. The cue uses only the selected bounded field set.
3. Non-fork prompts do not show the cue.
4. Forks with no meaningful bounded differences do not show noisy empty output.
5. No new panel, schema, or compare workflow is introduced.
6. Focused regression coverage protects the bounded behavior.

## Suggested test

One or two focused tests are enough.

Recommended shape:
- create a fork with a changed body and tags, then verify the rendered summary includes those field labels
- verify that a non-fork prompt or a fork with no bounded differences does not surface the cue

## Rollback

Rollback should be one isolated patch:
- revert the parent-difference summary helper
- revert the cue rendering
- remove the focused regression tests
- leave fork baseline reset and lineage naming improvements untouched

## Anti-goals

- do not add full diffs
- do not compare arbitrary version pairs
- do not add fork rationale metadata
- do not widen detail view into a comparison dashboard
- do not touch execution, analytics, chains, web search, sharing, or voice

## Notes for implementation

- Keep the slice boring.
- Changed-field labels are enough for v1.
- The operator needs quick selection confidence, not a mini Git diff.
