# PromptManager — Implementation Review

Date: 2026-04-12
Target: `Source-Matched Preview Priority v1`
Expected source: `docs/implementation-brief-2026-04-12-source-matched-preview-priority-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. It keeps the behavior inside the existing prompt preview/list seam, lets active plain-text search temporarily promote a credible `Source: ...` cue when that is the most helpful visible match, and avoids widening into ranking logic, provenance UI, or list redesign.

## What matches

### 1. The change stays in the intended seam
The implementation remains local to:
- `gui/prompt_preview.py`
- `gui/prompt_list_model.py`
- `tests/test_prompt_list_model.py`

That matches the brief's prompt-preview/list posture.

### 2. No-search preview priority stays intact
Without active search, the existing order remains:
- description
- scenario
- source

That preserves ordinary retrieval preview behavior exactly where the brief required stability.

### 3. Active search can promote a credible matching source cue
With active plain-text search, a credible `Source: ...` value can become the preview line when it matches the current search terms.
That is the intended v1 behavior and it uses the existing source-credibility seam rather than inventing a new provenance layer.

### 4. Weak or generic source values remain filtered out
Low-signal source values still do not get elevated into noisy preview text.
That keeps the slice aligned with the prior credible-source posture instead of widening into provenance noise.

### 5. UI refresh behavior is correctly covered
`PromptListModel.set_active_search_text()` now emits `PreviewRole` in addition to the existing match roles when the chosen preview can change.
That small wiring fix is justified and necessary, because otherwise the model logic could be correct while the visible preview line stayed stale after a search change.

### 6. Focused regression coverage exists
Focused tests cover:
- active-search source-priority selection
- no-search fallback stability
- weak/generic source guardrails
- preview-role emission when search changes the chosen preview
- existing prompt-list preview and highlight behavior still holding

Validation passed:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_list_model.py`
- result: `12 passed`

## What is missing

Nothing material relative to the brief.

A manual GUI pass for visual feel in a live prompt list was not added, but that is proportionate for this slice and not required to judge bounded alignment.

## What drifted / widened

No meaningful scope drift is visible.

One small implementation detail went slightly beyond the initial patch shape, but appropriately:
- `PromptListModel` now emits `PreviewRole` on active-search changes

That is not product-scope widening. It is the minimum correctness fix needed for the bounded preview-selection behavior to render reliably.

## What is unverified

### 1. Live visual usefulness in real browsing
This review did not include a manual GUI pass to judge whether operators notice the source-priority preview quickly enough in a dense result list.

### 2. Edge feel with multi-term ambiguous searches
Focused tests confirm the bounded model behavior, but this review did not manually inspect broader interactive search sessions with noisier operator input.

## Recommended next action

Treat `Source-Matched Preview Priority v1` as delivered.

Do not widen it into search-reason labels, provenance badges, or ranking explanation.
If another retrieval follow-up is needed later, it should be a separate tiny slice only after observing real remaining friction.

## Sources reviewed

- `docs/implementation-brief-2026-04-12-source-matched-preview-priority-v1.md`
- `gui/prompt_preview.py`
- `gui/prompt_list_model.py`
- `tests/test_prompt_list_model.py`
- focused test result: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_list_model.py` → `12 passed`
