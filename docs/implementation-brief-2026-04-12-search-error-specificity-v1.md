# PromptManager — Implementation Brief

Date: 2026-04-12
Status: delivered and spot-verified
Feature: Search Error Specificity v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `core/prompt_manager/search.py`
- `gui/prompt_list_presenter.py`

## Goal

Implement one bounded **Search Error Specificity** improvement so the prompt-search failure popup can show a useful backend/Chroma error message when available instead of only a generic failure line.

## Product intent

This slice strengthens the core loop at:
- retrieve,
- inspect.

It should reduce friction between:
- "search failed"
- and
- "I can immediately see what kind of backend/search problem happened."

The slice must stay quiet and local.
It must not turn PromptManager into a diagnostic console, maintenance workflow, or broad application-wide error-handling redesign.

## Scope

### In scope
- improve the existing prompt-search error reporting path only
- when backend search returns a useful concrete error, show that message in the existing error popup
- keep the current human-facing popup title behavior intact
- sanitize backend error text before showing it:
  - keep the useful message
  - avoid stack traces
  - avoid unbounded noisy dumps
- keep a safe fallback to the current generic message when no useful backend detail exists
- add focused regression coverage for specific backend error propagation and generic fallback behavior

### Out of scope
- special-casing only embedding dimension mismatch
- changing search logic, ranking, filtering, or similarity behavior
- changing embeddings, model configuration, or Chroma schema handling
- adding repair buttons, maintenance shortcuts, or auto-fix actions
- redesigning the whole app error system
- changing flows outside the current prompt-search error seam
- execution, analytics, chains, sharing, or voice work

## Recommended UX posture

Prefer one clear error detail over clever diagnosis.

Suggested v1:
- popup title stays something like `Unable to search prompts`
- popup body can show a concrete backend message when present, for example:
  - `Chroma error: Embedding dimension 1536 does not match collection dimension 3072.`
  - `Chroma error: no such column: collections.topic`
- if the backend detail is missing, empty, or too messy, fall back to the current generic message

Default recommendation:
- **show the real failure reason when you have it, but keep it readable**

## Likely implementation seam

### Search backend error shaping
- `core/prompt_manager/search.py`
  - preserve or expose useful backend error text instead of collapsing everything into an over-generic failure message
  - keep the change local to prompt-search failure handling

### GUI search error path
- `gui/prompt_list_presenter.py`
  - continue using the existing popup path
  - pass through the improved error message without widening into richer dialogs or maintenance hints

### Tests
- focused tests near the current search/presenter seam
- cover at least:
  - concrete backend error message shown to the caller
  - fallback to generic search error when detail is unavailable or unsuitable

## Happy-path scenario

1. User searches for prompts.
2. Backend search fails.
3. Popup still says search failed, but now also includes the concrete backend reason in readable form.
4. User can tell immediately whether this is, for example, an index mismatch, schema mismatch, or other backend issue.

That is enough for v1.

## Acceptance checks

1. Search failure popup can show a concrete backend/Chroma error message when one is available.
2. The shown detail is sanitized and readable, not a raw stack dump.
3. When no useful backend detail exists, the popup falls back to the current generic message.
4. No special-case lock-in to only one error class such as dimension mismatch.
5. Search logic, ranking, filtering, embeddings, and Chroma maintenance behavior remain unchanged.
6. No repair buttons, new dialogs, or maintenance workflows are introduced.
7. Focused regression coverage protects concrete-error and fallback behavior.

## Suggested test

Two focused tests are enough.

Recommended shape:
- a backend search failure with a concrete Chroma message reaches the user-visible error path in sanitized form
- a backend search failure without useful detail still shows the generic fallback message

If one golden concrete example is needed, use a dimension mismatch message as the regression sample, but do not bake the implementation into that one case only.

## Rollback

Rollback should be one isolated patch:
- remove the new backend-error formatting/preservation from the prompt-search seam
- remove the focused regression tests
- leave generic search failure behavior in place
- leave search logic, embeddings, and maintenance flows untouched

## Anti-goals

- do not build a full error taxonomy in v1
- do not special-case only embedding dimension mismatch
- do not add auto-fix or rebuild actions
- do not redesign the app-wide popup/error framework
- do not change the search algorithm or vector-store behavior
- do not widen into maintenance UX in this slice

## Notes for implementation

- Keep the slice boring.
- Prefer preserving the real backend message over inventing a guessed diagnosis.
- Sanitize for readability, not for cleverness.
- If the underlying exception detail is empty or ugly, prefer the safe generic fallback.

## Delivery note

Delivered in:
- `core/prompt_manager/search.py`
- `tests/test_prompt_manager_branches.py`

Focused validation:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_manager_branches.py -k 'surfaces_sanitized_backend_query_error or falls_back_for_unsuitable_backend_query_error'`
- result: `2 passed, 45 deselected`
