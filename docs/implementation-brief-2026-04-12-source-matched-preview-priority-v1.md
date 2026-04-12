# PromptManager — Implementation Brief

Date: 2026-04-12
Status: delivered and spot-verified
Feature: Source-Matched Preview Priority v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-12-search-match-highlight-v1.md`

## Goal

Implement one bounded **Source-Matched Preview Priority** improvement inside the existing main prompt list so the bounded preview line can better explain *why* a search result matches when the strongest visible match comes from stored source/provenance data.

## Product intent

This slice strengthens the core loop at:
- retrieve,
- inspect.

It should reduce friction between:
- "I searched and several results still look similar"
- and
- "I can see from the preview line that this result matches because of its source."

The slice must stay quiet and local.
It must not turn retrieval preview into a ranking system, explanation layer, or metadata-heavy list redesign.

## Scope

### In scope
- improve the existing preview-selection logic for the main prompt list only
- when active plain-text search matches the stored source cue, allow the preview line to prefer the source cue over the usual description/scenario-first order
- keep using only existing prompt data:
  - description
  - scenario
  - source
- keep the preview bounded to one compact secondary line
- keep search-match highlight behavior, ranking, filtering, selection, and list layout unchanged
- add focused regression coverage for active-search source-priority and ordinary no-search behavior

### Out of scope
- changing retrieval ranking or similarity logic
- changing search highlight rules
- adding new source badges, metadata rows, or explanation labels
- broad provenance taxonomy work
- new persistence or schema changes
- changes outside the current prompt-preview/list seam
- execution, analytics, chains, sharing, or voice work

## Recommended UX posture

Prefer one quiet source-priority adjustment over richer explanation.

Suggested v1:
- if search is inactive, keep the current preview priority unchanged
- if search is active and the source cue is the strongest visible search-aligned signal, let the preview line show that source cue instead of a less helpful generic description/scenario line
- if no credible source cue exists, keep the current preview behavior unchanged
- prefer no change over noisy or misleading preview switching

Default recommendation:
- **make the preview line more search-relevant, not more verbose**

## Likely implementation seam

### Preview selection
- `gui/prompt_preview.py`
  - extend the existing bounded preview helper with one active-search-aware source-priority rule
  - reuse the existing source-credibility and truncation helpers
  - keep the no-search default path intact

### Prompt-list model wiring
- `gui/prompt_list_model.py`
  - pass only the minimum active-search context needed for preview selection, if required
  - avoid introducing a second query source or broader preview state

### Tests
- `tests/test_prompt_list_model.py`
  - cover source cue becoming the chosen preview when active search matches source
  - cover ordinary no-search preview priority staying unchanged
  - cover weak/generic source values still staying out of preview selection

## Happy-path scenario

1. User searches for a source-like phrase such as a queue, notebook, or system label.
2. Several prompts still have similar titles.
3. A result with a credible matching source shows `Source: ...` in the preview line.
4. The operator can recognize that result faster without opening several prompts.

That is enough for v1.

## Acceptance checks

1. With active plain-text search, a credible matching source cue can become the chosen preview line.
2. With no active search, the current description/scenario/source preview priority remains unchanged.
3. Low-signal source values still do not surface as preview text.
4. Search highlight behavior remains unchanged and can operate on the chosen preview line as today.
5. Ranking, filtering, selection, and list layout remain unchanged.
6. No new persistence, schema, or extra list chrome is introduced.
7. Focused regression coverage protects source-priority and no-search fallback behavior.

## Suggested test

Two or three focused tests are enough.

Recommended shape:
- active search matching a credible source chooses `Source: ...` as the preview line
- no active search keeps description-first preview behavior
- generic/low-signal source still does not override the ordinary preview path

## Rollback

Rollback should be one isolated patch:
- remove the active-search-aware source-priority branch from preview selection
- remove any small model wiring added only for this behavior
- remove the focused regression tests
- leave retrieval preview, search highlight, ranking, filtering, and detail view behavior untouched

## Anti-goals

- do not change search ranking or similarity logic
- do not add search-reason labels or provenance badges
- do not add extra metadata rows to the prompt list
- do not widen into provenance-management work
- do not redesign the prompt list or preview delegate
- do not reopen Quick Capture cleanup in this slice

## Notes for implementation

- Keep the slice boring.
- Reuse the existing source-credibility helper instead of inventing new provenance rules.
- Prefer one active-search-aware preview choice over multiple competing heuristics.
- If the search-to-source signal is weak or ambiguous, keep the current preview priority unchanged.

## Delivery note

Delivered in:
- `gui/prompt_preview.py`
- `gui/prompt_list_model.py`
- `tests/test_prompt_list_model.py`

Focused validation:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_list_model.py`
- result: `12 passed`
