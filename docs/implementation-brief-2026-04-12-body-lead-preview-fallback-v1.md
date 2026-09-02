# PromptManager — Implementation Brief

Date: 2026-04-12
Status: ready
Feature: Body-Lead Preview Fallback v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-12-single-turn-user-prefix-strip-v1.md`

## Goal

Implement one bounded **Body-Lead Preview Fallback** improvement in the shared preview seam so prompts with weak or empty supporting metadata can still show one useful compact preview derived from the prompt body.

The intended v1 behavior is narrow:

> if `description`, `scenarios`, and credible `source` do not produce a good preview, use one short lead from `prompt.context`

This is **not** a summarization feature.
It is one deterministic fallback for already-stored prompt text.

## Product intent

This slice strengthens the core loop at:
- retrieve,
- inspect,
- reuse.

It should reduce friction between:
- "I found a prompt row or a similar-match row with almost no helpful preview text"
- and
- "I can immediately judge what this prompt is about from one compact visible cue."

## Why this slice is earned now

PromptManager already has a shared bounded preview seam used by retrieval surfaces.
That seam currently prefers:
- matching credible source cue during active search,
- description,
- scenarios,
- source cue.

The remaining visible miss is simple and local:
- some prompts still have a useful body,
- but weak or empty description/scenario/source metadata,
- so they end up with little or no preview help.

A body-lead fallback improves that without opening a new surface and without widening into search ranking, summarization, or prompt cleanup.

## Scope

### In scope
- one deterministic shared preview fallback derived from `prompt.context`
- apply it only after current stronger preview candidates fail
- keep preview single-line and truncated using the existing shared preview helpers
- reuse the same seam for retrieval/list-style surfaces already consuming shared preview text
- add focused regression coverage for body-fallback and unchanged higher-priority cases

### Out of scope
- LLM summarization
- multi-line previews
- search ranking changes
- new UI controls, labels, or panels
- Quick Capture cleanup changes
- detail-view layout changes
- transcript parsing
- new metadata fields or schema changes
- broader preview-policy redesign

## Recommended UX posture

Prefer one quiet fallback only when stronger preview sources are absent.

Suggested v1:
- if a prompt already has a good description preview, keep it
- if a scenario already provides the first credible preview, keep it
- if a credible source cue is the best current fallback, keep it
- only then derive one compact preview from the opening body text

Reason:
- it stays inside the existing preview contract
- it improves retrieval confidence without adding interface noise
- it keeps body-derived preview as fallback, not as the new primary policy

## Data source

Use only existing prompt data already present in memory:
- `prompt.description`
- `prompt.scenarios`
- `prompt.source`
- `prompt.context`

Do not add storage, caching, or new persistence in this slice.

## Likely implementation seam

### Shared preview helper
- `gui/prompt_preview.py`
  - extend `build_prompt_preview()` with one final body-lead fallback
  - flatten and truncate the body lead through the existing helper path
  - ignore empty or low-signal body text

### Consumers protected by existing seam
- `gui/prompt_list_model.py`
  - expected to benefit through the existing `PreviewRole`
- `gui/dialogs/draft_promote.py`
  - expected to benefit indirectly where similar-match labels reuse `build_prompt_preview()`

### Tests
- `tests/test_prompt_list_model.py`
  - add focused coverage for body-fallback and unchanged current priorities
- `tests/test_draft_promote_dialog.py`
  - add one focused case showing similar-match label gains a bounded body preview when description is absent

## Happy-path scenario

1. A saved prompt has:
   - empty description,
   - no useful scenarios,
   - no credible source cue,
   - body: `Summarize deployment risks for this release and call out rollback concerns.`
2. The prompt appears in the main list or similar-match advisory surface.
3. The preview shows one compact lead derived from that body.
4. The operator can judge prompt fit faster without opening detail view first.

That is enough for v1.

## Acceptance checks

1. A prompt with empty description, empty scenarios, weak source, and credible body shows a compact preview derived from the body lead.
2. A prompt with a strong description keeps the existing description-first preview behavior unchanged.
3. A prompt with a credible matching source cue during active plain-text search keeps the existing source-priority search behavior unchanged.
4. A prompt with a credible scenario fallback keeps the existing scenario-first fallback unchanged when description is absent.
5. Empty, whitespace-only, or too-weak body text does not produce a body preview.
6. Similar-match rows that rely on `build_prompt_preview()` can show the same bounded body fallback when description is absent.
7. No new UI controls, schema changes, search-ranking changes, or Quick Capture behavior changes are introduced.

## Suggested test set

A small deterministic set is enough:
- body fallback appears when description/scenarios/source are not useful
- description still wins over body fallback
- active-search matching credible source cue still wins when applicable
- weak body does not create preview text
- similar-match row can show body-derived preview when no better preview source exists

## Rollback

Rollback should be one isolated patch:
- remove the body-lead fallback from `build_prompt_preview()`
- remove focused regression tests tied to the new fallback
- leave existing preview priority and consuming surfaces untouched

## Anti-goals

- do not summarize prompt bodies
- do not parse structure from the body
- do not change search result ordering
- do not add preview expansion or hover cards
- do not widen into detail-view rewrite
- do not reopen Quick Capture cleanup work from this slice
- do not turn preview helpers into a ranking or scoring subsystem

## Notes for implementation

- Keep the slice boring.
- Keep body fallback last in priority order.
- Prefer no preview over low-signal filler preview.
- Reuse the existing flatten/truncate helpers instead of inventing new formatting logic.
- If implementation pressure starts changing ranking, labels, or layout, the slice is drifting.
