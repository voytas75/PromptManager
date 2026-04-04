# PromptManager — Next Slice Brief

Date: 2026-04-04
Status: delivered
Slice name: Recent Reopen
Primary source: `docs/product-boundary-ssot.md`
Supporting sources:
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/next-slice-brief-2026-04-04-quick-capture-to-draft.md`

## Recommended slice

Add one **Recent Reopen** happy-path workflow that lets the user quickly reopen a recently touched prompt from the existing catalog without re-searching for it.

This slice should expose one compact UI surface for recent prompts and let the user jump straight back into the existing detail/editor flow.

## Why this now

The previous slice improved **capture**.
The next highest-value small move is to reduce friction in **retrieve → inspect → reuse**.

The point is simple:

> if a prompt was just created, edited, or reused recently, it should be trivial to get back to it.

This directly strengthens the product center without widening into analytics, chains, web enrichment, or assistant behavior.

## User problem solved

Right now, a user may already have the right prompt in the catalog but still has to:
- remember its name,
- search again,
- scan results,
- click back into it.

For recently used prompts, that is unnecessary friction.

This slice should make the first reuse step almost boring:

> open recent list → pick prompt → land in detail/editor flow

## Boundaries

### Do now
Implement exactly one narrow end-to-end workflow:

1. Add one compact **Recent** entry point in the existing GUI.
2. Show a short list of recently touched prompts from existing data already available in the catalog.
3. Limit v1 to one deterministic ordering, e.g. newest by `last_modified`.
4. Let the user select one item and open it in the existing detail flow.
5. If natural in the current UI seam, allow one-click edit from that same recent selection.
6. Add one deterministic test for the ordering / selection happy path.

### Strong defaults for v1
- Prefer existing prompt fields (`last_modified`, existing prompt list/state) over adding new persistence.
- Prefer a small menu/dialog/panel over a new major screen.
- Prefer a short fixed list, e.g. top 5 or top 10, over configurability.
- Prefer reopening over richer analytics or usage scoring.

## Do later
- pinned prompts
- favorites
- usage-weighted ranking
- “recent by execution” versus “recent by edit” split
- session-specific recents
- cross-surface command palette enhancements
- keyboard-first MRU navigation
- advanced reopen filters

## Acceptance checks

1. A user can access a compact recent-prompts surface from one obvious GUI entry point.
2. The list is deterministic and uses existing prompt data.
3. Selecting a recent prompt opens the existing detail/editor flow without requiring search.
4. The slice works without LiteLLM, Redis, web search, chains, or external services.
5. One deterministic regression test protects the happy path.

## Rollback

Rollback should be one isolated patch:
- remove the Recent entry point,
- remove the narrow recent-list seam,
- remove the focused regression test,
- leave the rest of the catalog/editor/runtime untouched.

## Anti-goals

- Do not redesign global navigation.
- Do not add favorites, pinning, or collections in this slice.
- Do not introduce new usage analytics or scoring systems.
- Do not widen into command-palette overhaul.
- Do not touch chains, sharing, voice, or web-enrichment behavior.
- Do not add new persistence unless absolutely required.

## Suggested implementation posture

Keep the slice small and honest:
- one entry point,
- one recent list,
- one reopen action,
- one deterministic test,
- minimal doc update if needed.

The product win is not cleverness.
The product win is reducing the friction between **I used this prompt recently** and **I am back inside it now**.

## Definition of done

The slice is done when:
- the user can reopen a recent prompt in one short flow,
- the reopened prompt is visible immediately in the existing UI,
- the implementation stays bounded,
- the test passes,
- nothing peripheral was expanded.

## Implementation note

Delivered on 2026-04-04 as a compact recent-prompts dialog opened from the main toolbar.
Ordering is deterministic via existing `last_modified` data with stable tie-breakers, and selection routes back into the existing detail flow without adding new persistence.
