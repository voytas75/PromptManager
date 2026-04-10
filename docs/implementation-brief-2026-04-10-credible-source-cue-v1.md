# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Credible Source Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded **Credible Source Cue** improvement so the existing detail view shows source/provenance only when the stored source is actually useful to the operator, while hiding low-signal technical markers that do not help prompt selection or inspection.

## Product intent

This slice strengthens the core loop at:
- inspect,
- retrieve.

It should reduce friction between:
- "I am looking at prompt provenance"
- and
- "this source information actually helps me judge what this prompt is and where it came from."

## Scope

### In scope
- one bounded credibility filter for source rendering in the shared detail widget
- reuse the same low-signal source logic already used by retrieval preview paths
- flatten noisy source text when it is shown
- keep draft-state cues and last-modified cues unchanged
- focused regression coverage for:
  - credible source remains visible
  - low-signal source stays hidden
  - existing draft cue still renders

### Out of scope
- new provenance fields, presets, or taxonomy
- provenance filtering/search UI
- new panels, chips, or metadata sections
- analytics or source dashboards
- changes to capture storage semantics
- retrieval ranking changes beyond shared credibility logic reuse

## Recommended UX posture

Prefer one quiet consistency pass.

Suggested v1:
- if source is credible, keep showing it in the existing inspection cue line
- if source is low-signal, do not show it as a source cue
- examples of low-signal values:
  - `quick_capture`
  - `local`
  - `unknown`
- keep `Draft (quick_capture)` intact when draft metadata exists, because that is status rather than user-facing provenance

Default recommendation:
- **filter low-value provenance noise**, not add more provenance chrome

Reason:
- smallest useful improvement
- aligns detail behavior with retrieval-preview behavior
- makes existing provenance more trustworthy by removing junk signals

## Data source

Use existing prompt data only.

Relevant fields:
- `prompt.source`
- `prompt.ext2.capture_state`
- `prompt.ext2.capture_method`

Do not add new persistence in this slice.

## Likely implementation seam

### Shared provenance helper
- `gui/prompt_preview.py`
  - expose one small helper for credible source cue building
  - keep low-signal filtering centralized

### Detail view
- `gui/widgets/prompt_detail_widget.py`
  - reuse the shared helper when assembling inspection cues
  - preserve current draft and timestamp behavior

### Tests
- `tests/test_prompt_detail_widget.py`
- `tests/test_prompt_list_model.py` only if the shared helper change needs direct confidence on retrieval-preview behavior

## Happy-path scenario

1. User opens a prompt in detail view.
2. Prompt has a meaningful source like `chat thread` or `ops notebook`.
3. Detail view shows that source in the existing inspection cue line.
4. Another prompt has only a technical marker like `quick_capture` or `local` in `source`.
5. Detail view hides that low-signal source instead of pretending it is useful provenance.
6. Draft-state cue still works when present.

That is enough for v1.

## Acceptance checks

1. Credible source values remain visible in the existing inspection cue line.
2. Low-signal source values do not render as a source cue.
3. Draft-state cue still renders when applicable.
4. Last-modified cue remains unchanged.
5. No new UI surface or schema is introduced.
6. Focused regression coverage protects the bounded rendering behavior.

## Suggested test

One or two focused widget tests are enough.

Recommended shape:
- one prompt with `source="ops notebook"` and verify source stays visible
- one prompt with `source="quick_capture"` or `source="local"` and verify source does not render in inspection cues while other cues still do

## Rollback

Rollback should be one isolated patch:
- remove the shared credible-source helper if introduced solely for this slice
- revert detail-widget source filtering to the previous direct rendering behavior
- remove the focused regression tests
- leave the rest of inspect flow untouched

## Anti-goals

- do not add provenance badges or a source panel
- do not redesign the inspection line
- do not change capture-state semantics
- do not widen this into provenance search/filter work
- do not add analytics or taxonomy work

## Notes for implementation

- Keep the slice boring.
- Prefer consistency with existing retrieval preview logic over inventing a second provenance rule set.
- Better provenance means clearer provenance, not more provenance.
