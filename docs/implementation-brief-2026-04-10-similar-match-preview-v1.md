# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Similar Match Preview v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/next-slice-brief-2026-04-04-promote-similar-check-v1.md`

## Goal

Implement one bounded **Similar Match Preview** improvement inside the existing **Draft Promote** advisory flow so the operator can judge similar existing prompts faster before deciding whether to promote the draft as a new asset.

## Product intent

This slice strengthens the core loop at:
- normalize,
- inspect,
- refine.

It should reduce friction between:
- “I see similar prompts exist”
- and
- “I understand quickly whether one of them is probably the asset I meant.”

## Scope

### In scope
- one compact identifying cue for each similar prompt already shown in the Draft Promote advisory list
- cue derived only from existing prompt data
- deterministic preview selection and truncation
- keeping the current advisory summary and current two operator actions intact
- one focused deterministic regression test path, or a small extension of the current promote-dialog tests

### Out of scope
- full duplicate detection redesign
- auto-merge, auto-block, or replace-existing behavior
- batch import duplicate handling
- new persistence fields or schema changes
- AI-generated summaries
- search ranking changes
- prompt list redesign outside the existing Draft Promote advisory surface
- chains, analytics, sharing, voice, or web-enriched execution work

## Recommended UX posture

Prefer a small and quiet upgrade to the current advisory list.

Suggested v1:
- keep the existing similar-prompt list and the same operator actions
- extend each similar match with one compact distinguishing cue using existing data
- prefer the same kind of signal already proven useful in Retrieval Preview v1:
  - description
  - scenario text
  - credible source
- show the cue only when it is short and credible
- if the signal is weak, keep the current row clean

Default recommendation:
- **one bounded secondary cue**, not badges, chips, cards, or an expanded review panel

Reason:
- lower scope
- consistent with the product boundary
- improves the exact decision moment that matters without creating a second ingest workflow

## Data source

Use existing prompt data only.

Preferred signal order:
1. description
2. scenario text
3. credible source

Preferred formatting posture:
- flatten multiline text into one readable line
- deterministically truncate long cues
- ignore low-signal placeholder values

Do not add new persistence in this slice.

## Likely implementation seam

### Existing advisory surface
- `gui/dialogs/draft_promote.py`
  - currently renders the similar-prompt list
  - already supports advisory summary, selection, and open-existing flow
  - currently uses a compact title/category label plus tooltip metadata

### Shared preview logic
Prefer one shared helper rather than two diverging heuristics.

Strong preference:
- extract or reuse the bounded preview-selection logic already used for Retrieval Preview v1
- keep the extraction minimal and local to the shared retrieval-preview concern

Possible seam:
- move preview-building helper into a small shared module or keep a narrow helper that both:
  - `gui/prompt_list_model.py`
  - `gui/dialogs/draft_promote.py`
  can consume without widening unrelated contracts

### Tests
- extend `tests/test_draft_promote_dialog.py`
- optionally add one narrow helper test only if the shared extraction needs direct coverage

## Happy-path scenario

1. User opens Promote Draft for a captured prompt.
2. Advisory similar-prompt list appears.
3. Each similar result still shows the current identifying label.
4. When an existing prompt has a short credible distinguishing signal, one compact preview cue is shown as part of the visible advisory item.
5. User can more easily decide whether to:
   - open similar existing
   - or continue promoting as new.

That is enough for v1.

## Acceptance checks

1. The Draft Promote advisory list can show one compact distinguishing cue for similar existing prompts when sufficient existing signal is present.
2. The cue is derived only from existing prompt data.
3. Similar prompts without credible signal remain clean rather than showing noisy filler.
4. The current advisory flow and current operator actions remain unchanged.
5. The slice works without LiteLLM, Redis, web search, chains, or external services.
6. Focused regression coverage protects:
   - cue visible when signal exists
   - cue absent when signal is weak
   - bounded truncation/formatting in the advisory surface

## Suggested test

One focused test extension is enough.

Recommended shape:
- construct one similar prompt with a useful description or scenario
- open the existing Draft Promote dialog with that prompt in the advisory list
- verify the visible advisory item exposes the bounded cue
- verify a weak-signal prompt still uses the clean fallback presentation

## Rollback

Rollback should be one isolated patch:
- remove the new similar-match cue from the Draft Promote advisory list
- remove the minimal shared helper extraction if one was introduced solely for this slice
- remove the focused regression test additions
- leave Retrieval Preview v1, Draft Promote, and the current advisory actions untouched

## Anti-goals

- do not redesign the whole Draft Promote dialog
- do not add a second duplicate review screen
- do not introduce AI summarization
- do not add replace-existing, merge, or update-existing workflows
- do not widen this into batch import or broader ingest orchestration
- do not touch retrieval ranking or prompt list behavior outside this advisory seam

## Notes for implementation

- Keep the slice boring.
- Reuse the retrieval-preview posture rather than inventing a second preview language.
- Improve decision confidence, not feature count.
- If shared helper extraction starts to widen beyond the two current surfaces, stop and keep the implementation local.
