# PromptManager — Implementation Brief

Date: 2026-04-12
Status: ready
Feature: Favorites v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `gui/widgets/prompt_filter_panel.py`
- `gui/widgets/prompt_detail_widget.py`
- `models/prompt_model.py`
- `core/repository/prompts.py`

## Goal

Implement one bounded **Favorites v1** slice so the operator can mark a prompt as a favorite and then retrieve favorite prompts faster without relying on repeated search.

The desired experience is simple:

> this is a prompt I return to often -> mark it as favorite -> show me favorites quickly later

This slice should strengthen retrieval and reuse speed without widening into collections, saved sets, pinning systems, or recommendation logic.

## Product intent

This slice strengthens the product near the core loop at:
- retrieve,
- inspect,
- reuse.

PromptManager already supports:
- search,
- recent reopen,
- inspection cues,
- quick reuse.

This slice should add one lightweight explicit operator signal:

> I want to keep this prompt close at hand

That signal should stay small, local, and single-user in character.

## Scope

### In scope
- add one prompt-level favorite state
- let the operator toggle favorite state from the prompt detail flow
- add one simple favorites-only retrieval/filter path in the existing prompt list/filter surface
- persist favorite state using the current prompt persistence path with the smallest reasonable implementation approach
- keep the slice local to prompt storage, detail action, and list filtering
- add focused regression coverage for:
  - persistence/hydration of favorite state
  - detail-view toggle behavior
  - favorites-only filtering behavior

### Out of scope
- collections
- saved sets
- folders
- pinning system beyond the single favorite flag
- ranking/recommendation logic based on favorites
- bulk favorite management
- favorite badges across every surface
- new dialogs or management panels
- broad filter-panel redesign
- sharing, analytics, chains, or workspace changes

## Recommended UX posture

Prefer **one explicit favorite signal** over a richer organization system.

Suggested v1 posture:
- one favorite toggle in the prompt detail surface
- one simple favorites-only filter path in the existing list/filter flow
- no hierarchy, no grouping, no extra management UI

Default recommendation:
- **make frequent prompts easier to reopen, not more administratively complex**

## Data/persistence posture

Prefer the smallest durable implementation.

Recommended v1:
- use the current prompt persistence path with a lightweight stored favorite flag
- avoid introducing a new subsystem or broad schema redesign if the existing prompt metadata seam can carry the state cleanly
- keep the favorite state prompt-local and deterministic

If a broader schema change starts to appear necessary, stop and reassess whether the first slice can stay smaller.

## Likely implementation seams

### Prompt data / persistence
- `models/prompt_model.py`
- `core/repository/prompts.py`

### Detail action seam
- `gui/widgets/prompt_detail_widget.py`

### Prompt list / filter seam
- `gui/widgets/prompt_filter_panel.py`
- plus the smallest presenter/controller seam needed to make the filter effective

### Tests
Likely focused seams:
- prompt model / repository tests for stored favorite state
- `tests/test_prompt_detail_widget.py`
- prompt list/filter tests where existing filtering behavior is already covered

## Happy-path scenarios

### Scenario A: mark a prompt as favorite
1. User opens a reusable prompt in the detail view.
2. User marks it as favorite.
3. The prompt keeps that state after refresh/reload.

### Scenario B: retrieve favorites quickly
1. User returns later to the prompt catalog.
2. User enables the favorites-only filter.
3. Favorite prompts appear without needing search terms.

### Scenario C: remove from favorites
1. User opens a favorited prompt.
2. User removes the favorite mark.
3. The prompt no longer appears in the favorites-only filtered list.

That is enough for v1.

## Acceptance checks

1. A prompt can be marked and unmarked as favorite from the detail flow.
2. Favorite state persists through the existing prompt storage path.
3. The prompt list/filter surface can show favorites only.
4. Plain non-favorite prompts are excluded when the favorites-only filter is active.
5. No collections, folders, pinning system, or management panel is introduced.
6. Focused regression coverage protects stored state, toggle behavior, and favorites-only filtering.
7. The slice stays local and does not redesign retrieval or organization broadly.

## Rollback

Rollback should be one isolated patch:
- remove the favorite state handling,
- remove the detail toggle,
- remove the favorites-only filter path,
- remove the focused regression tests,
- leave ordinary search/recent/detail/reuse behavior untouched.

## Anti-goals

- do not widen into collections or saved sets
- do not add recommendation logic from favorites
- do not redesign the whole filter panel
- do not add bulk-organize surfaces
- do not turn favorites into pinning-plus-history-plus-ranking in one slice
- do not bundle another product slice into the same patch

## Notes for implementation

- Keep the slice boring.
- The user need is simple: a few prompts should be easier to get back to.
- If implementation starts adding taxonomy or curation mechanics, the slice is drifting.
- The first version should optimize for clarity and speed, not organizational power.
