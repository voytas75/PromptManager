# PromptManager — Implementation Brief

Date: 2026-04-12
Status: delivered and spot-verified
Feature: Search Match Highlight v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/next-slice-brief-2026-04-11-search-match-highlight-v1.md`

## Goal

Implement one bounded **Search Match Highlight** improvement inside the existing main prompt list so the operator can spot why a search result matches before opening several similar prompts one by one.

## Product intent

This slice strengthens the core loop at:
- retrieve,
- inspect.

It should reduce friction between:
- "I searched for something"
- and
- "I can see which result is likely right from the list itself."

The slice must stay quiet and local.
It must not turn the prompt list into a search-explanation system, ranking surface, or broader redesign.

## Scope

### In scope
- add one subtle search-match highlight treatment to the existing prompt-list row text
- support highlighting inside:
  - the title line
  - the existing preview line, when present
- activate highlighting only when a normal text search is currently active
- keep the match logic deterministic and boring:
  - case-insensitive
  - plain text token or substring matching only
- keep current ranking, filtering, selection, and list layout behavior unchanged
- add focused regression coverage for active-search and no-search states

### Out of scope
- changing search ranking or similarity logic
- semantic or score-based match explanations
- badges, chips, or extra metadata rows
- broader prompt-list redesign
- new persistence or schema changes
- changes outside the current prompt-list search/rendering seam
- execution, analytics, chains, sharing, or voice work

## Recommended UX posture

Prefer one quiet highlight cue over richer search explanation.

Suggested v1:
- if search text is active, subtly emphasize matching text already visible in the row
- keep the emphasis readable but secondary to selection state
- if search text is empty, render the list exactly as it does today
- if the match logic cannot confidently highlight a fragment, prefer no highlight over noisy or misleading emphasis

Default recommendation:
- **show just enough to guide the eye, not enough to explain the search system**

## Likely implementation seam

### Search state / list wiring
- `gui/prompt_search_controller.py`
  - confirm or extend the current active-search seam only if minimal wiring is needed
- list presenter / list view seam already fed by toolbar search state
  - reuse the existing current-search flow rather than creating a second query source

### Prompt-list rendering
- `gui/prompt_list_delegate.py`
  - extend row rendering to support one subtle active-search match emphasis in title and preview text
  - keep the current two-line list structure intact
- `gui/prompt_list_model.py`
  - expose only the minimum additional role/state needed for rendering, if any

### Tests
- `tests/test_prompt_list_model.py`
  - extend with focused coverage for active-search and no-search states
- add one delegate-focused test if needed for the bounded highlight rendering path

## Happy-path scenario

1. User types a normal text search.
2. Prompt list narrows as it does today.
3. Matching text in the title and, when present, preview line becomes subtly highlighted.
4. User can distinguish similar results faster without opening several prompts.

That is enough for v1.

## Acceptance checks

1. With active text search, matching query text is subtly highlighted in the prompt-list title when present.
2. With active text search and an existing preview line, matching preview text can also be highlighted.
3. With no active search, prompt-list rendering stays unchanged.
4. Sorting, filtering, ranking, and selection behavior remain unchanged.
5. No new persistence, schema, or search-index fields are introduced.
6. The implementation stays inside the current prompt-list search/rendering seam.
7. Focused regression coverage protects active-search and no-search behavior.

## Suggested test

Two or three focused tests are enough.

Recommended shape:
- active search with a title match shows highlighted title text
- active search with a preview match shows highlighted preview text
- empty search leaves rendering/data in the current plain state

## Rollback

Rollback should be one isolated patch:
- remove the search-match highlight rendering from the list seam
- remove any small search-state wiring added only for this feature
- remove the focused regression tests
- leave retrieval preview, filtering, ranking, and detail view behavior untouched

## Anti-goals

- do not redesign the prompt list
- do not change search ranking or retrieval logic
- do not add semantic-match explanations
- do not add extra badges, icons, or metadata rows
- do not widen into execution, analytics, chains, sharing, or voice
- do not reopen Quick Capture cleanup in this slice

## Notes for implementation

- Keep the slice boring.
- Reuse the existing toolbar-search/list-rendering seam.
- Prefer one subtle emphasis style that behaves cleanly under row selection.
- If exact fragment rendering becomes costly or messy, keep the first cut minimal rather than broadening the delegate architecture.

## Delivery note

Delivered in:
- `gui/prompt_list_model.py`
- `gui/prompt_list_delegate.py`
- `gui/prompt_list_presenter.py`
- `tests/test_prompt_list_model.py`

Focused validation:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_list_model.py`
- result: `8 passed`
