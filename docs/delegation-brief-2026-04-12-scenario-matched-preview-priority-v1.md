# PromptManager — Delegation Brief

Date: 2026-04-12
Status: ready for delegation
Feature: Scenario-Matched Preview Priority v1
Implementation brief:
- `docs/implementation-brief-2026-04-12-scenario-matched-preview-priority-v1.md`

## Mission

Implement the bounded slice described in the implementation brief:
- add one active-search-aware scenario-priority branch in the shared preview seam
- after the existing source-match check, keep the description when it matches any active search term
- otherwise allow the first credible scenario in stored order that matches any active search term to become the preview line
- preserve no-search behavior
- preserve existing source-match priority
- keep the slice small, local, and deterministic

This is a small adjacent-core retrieval slice.
Do not widen it.

## Scope

### In scope
- extend the shared preview helper so active plain-text search can prefer the first credible matching scenario in stored order over a non-matching generic description
- keep the description when it matches any active search term
- keep ordinary no-search preview selection unchanged
- keep existing active-search source-match priority unchanged
- add focused deterministic tests for the new branch and unchanged priority behavior
- touch only the minimum files needed to ship the slice cleanly

### Out of scope
- ranking changes
- filtering changes
- semantic search changes
- Quick Capture changes
- Draft Promote changes
- detail-view changes
- new UI controls or labels
- metadata/schema changes
- broad preview-policy cleanup
- commit or push

## Expected implementation seam

Primary seam:
- `gui/prompt_preview.py`

Likely focused tests:
- `tests/test_prompt_list_model.py`

Touch adjacent files only if required by the existing seam.

## Acceptance checks

1. With active plain-text search, after the existing source-match check, a first credible scenario in stored order can become the preview line when the description does not match any active search term.
2. With no active search, current preview order stays unchanged.
3. Existing active-search source-match priority stays unchanged.
4. A matching description remains the preview when it matches any active search term.
5. Search highlight continues to work on the chosen preview line.
6. No ranking, filtering, selection, layout, schema, or persistence behavior changes are introduced.
7. Focused regression coverage protects the new branch and unchanged baseline.

## Validation

Run focused validation only.
Suggested target set:
- `pytest -q tests/test_prompt_list_model.py`

If one narrowly adjacent test becomes necessary because of the exact seam touched, include it, but keep validation focused.

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

- prefer boring implementation over clever preview scoring
- preserve current no-search behavior exactly
- preserve existing source-match priority exactly
- reuse the existing credibility helper rather than inventing a new credibility policy
- keep the search-aware override narrow and explicit
- if implementation pressure starts affecting ranking, layout, or unrelated consumers, stop and keep the slice narrower
