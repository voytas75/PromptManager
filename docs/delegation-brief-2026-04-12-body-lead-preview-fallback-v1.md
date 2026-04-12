# PromptManager — Delegation Brief

Date: 2026-04-12
Status: ready for delegation
Feature: Body-Lead Preview Fallback v1
Implementation brief:
- `docs/implementation-brief-2026-04-12-body-lead-preview-fallback-v1.md`

## Mission

Implement the bounded slice described in the implementation brief:
- add one final shared-preview fallback derived from the prompt body lead
- keep it deterministic, single-line, and low-noise
- preserve current stronger preview priorities and existing UX posture

This is a small adjacent-core slice.
Do not widen it.

## Scope

### In scope
- extend the shared preview helper so `prompt.context` can provide one compact fallback preview only when stronger preview sources fail
- preserve existing preview priority for:
  - active-search matching credible source cue
  - description
  - scenarios
  - credible source cue
- add focused deterministic tests for body fallback and unchanged priority behavior
- touch only the minimum files required to ship the slice cleanly

### Out of scope
- summarization
- search ranking changes
- multi-line previews
- new UI controls or labels
- Quick Capture changes
- detail-view redesign
- metadata/schema changes
- broad preview-policy cleanup
- commit or push

## Expected implementation seam

Primary seam:
- `gui/prompt_preview.py`

Likely focused tests:
- `tests/test_prompt_list_model.py`
- `tests/test_draft_promote_dialog.py`

Touch adjacent files only if required by the existing seam.

## Acceptance checks

1. A prompt with empty description, empty scenarios, weak source, and credible body shows a compact preview derived from the body lead.
2. A prompt with a strong description keeps existing description-first preview behavior unchanged.
3. A prompt with a credible matching source cue during active plain-text search keeps the existing source-priority search behavior unchanged.
4. A prompt with a credible scenario fallback keeps the existing scenario-first fallback unchanged when description is absent.
5. Empty, whitespace-only, or too-weak body text does not produce a body preview.
6. Similar-match rows that rely on shared preview helpers can show the same bounded body fallback when description is absent.
7. No new UI controls, schema changes, search-ranking changes, or Quick Capture behavior changes are introduced.

## Validation

Run focused validation only.
Suggested target set:
- `pytest -q tests/test_prompt_list_model.py tests/test_draft_promote_dialog.py`

If another narrowly adjacent test file becomes necessary because of the exact seam touched, include it, but keep validation focused.

## Deliverable

When done, report only:
1. what changed
2. files changed
3. validation run + result
4. whether the slice stayed bounded

Do not commit.
Do not push.
Do not do extra docs cleanup beyond what is necessary to keep the slice coherent.

## Guardrails

- prefer boring implementation over clever preview logic
- prefer no preview over low-signal filler
- preserve existing priority order unless the brief explicitly changes it
- if implementation pressure starts affecting ranking, layout, or unrelated surfaces, stop and keep the slice narrower
