# PromptManager — Implementation Brief

Date: 2026-04-04
Status: ready-for-delegation
Feature: Recent Reopen
Primary source: `docs/next-slice-brief-2026-04-04-recent-reopen.md`

## Goal

Implement one bounded **Recent Reopen** flow that lets the user quickly reopen a recently touched prompt from the existing catalog without re-running search.

## Product intent

This slice strengthens the core loop at:
- retrieve,
- inspect,
- reuse.

It should reduce friction between:
- “I used or edited this prompt recently”
- and
- “I am back inside it now.”

## Scope

### In scope
- one compact GUI entry point for recent prompts
- deterministic recent ordering using existing prompt data
- reopen into the existing detail flow
- optional handoff into editor only if it is essentially free in the current seam
- one deterministic regression test

### Out of scope
- favorites
- pinned prompts
- analytics or usage scoring
- command palette overhaul
- new persistence layer
- chains, sharing, voice, web enrichment
- broad navigation redesign

## Recommended UX posture

Prefer a small and quiet surface.

Suggested v1:
- add `Recent Prompts…` as a menu action in the existing toolbar area
- open a compact dialog listing a small number of recently touched prompts
- selecting a prompt should reopen it in the existing detail flow

Default recommendation:
- **detail-first**, not auto-edit-first

Reason:
- smaller UX commitment
- lower surprise cost
- easier to validate as a bounded retrieval/reuse improvement

## Data source

Use existing prompt data only.

Preferred ordering:
- `last_modified DESC`

Preferred v1 list size:
- top 5 or top 10

Do not add new persistence for “recent” in this slice.

## Likely implementation seam

### UI
- `gui/dialogs/recent_prompts.py`
  - compact dialog
  - list of prompts
  - returns selected prompt or prompt id

### Toolbar / wiring
- `gui/widgets/prompt_toolbar.py`
  - add `recent_requested` signal
  - add `Recent Prompts…` action in existing menu area

- `gui/main_view_builder.py`
  - connect `recent_requested`

- `gui/main_view_callbacks_factory.py`
  - map `recent_requested` to prompt actions handler

### Action handling
- `gui/main_window_handlers.py`
  - add `open_recent_prompts()`
  - fetch prompt list from existing in-memory/model-facing seam
  - sort by `last_modified DESC`
  - trim to bounded list size
  - open dialog
  - on selection: show selected prompt in existing detail flow

## Preferred source of prompt list

Prefer the nearest existing UI-facing source over adding manager/repository complexity.

Strong preference:
- reuse existing prompt list/model state already available to prompt actions or presenter layer

Do not widen backend contracts unless absolutely required.

## Happy-path scenario

1. User opens `Recent Prompts…`
2. Dialog shows top N prompts sorted by `last_modified DESC`
3. User chooses one
4. That prompt becomes selected in the existing UI
5. Existing detail panel displays it immediately

That is enough for v1.

## Acceptance checks

1. A user can access a compact recent-prompts surface from one obvious GUI entry point.
2. The recent list is deterministic and uses existing prompt data.
3. Selecting a recent prompt opens it in the existing detail flow without requiring search.
4. The slice works without LiteLLM, Redis, web search, chains, or external services.
5. One deterministic regression test protects the happy path.

## Suggested test

One focused test is enough.

Recommended shape:
- provide a small set of prompts with distinct `last_modified` values
- verify descending ordering in the recent list seam
- simulate one selection
- verify the selected prompt is routed into the existing select/detail flow

## Rollback

Rollback should be one isolated patch:
- remove the Recent entry point
- remove the recent dialog
- remove the narrow reopen seam
- remove the focused regression test
- leave the rest of the catalog/editor/runtime untouched

## Anti-goals

- do not redesign search
- do not add favorites or pinning
- do not add execution-based usage ranking
- do not create a second navigation system
- do not add persistent MRU storage
- do not widen into productivity/dashboard work

## Notes for implementation

- Keep the list short and deterministic.
- Prefer detail-first reopen.
- Do not over-model this.
- Use the smallest seam that makes “recent reopen” real.
