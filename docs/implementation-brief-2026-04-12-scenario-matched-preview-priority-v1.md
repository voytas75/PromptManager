# PromptManager — Implementation Brief

Date: 2026-04-12
Status: ready
Feature: Scenario-Matched Preview Priority v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-12-body-lead-preview-fallback-v1.md`

## Goal

Implement one bounded **Scenario-Matched Preview Priority** improvement in the shared preview seam so an active plain-text search can prefer a matching credible scenario cue over a non-matching generic description.

The intended v1 behavior is narrow:

> when active search is present, keep the existing source-match override first; otherwise, if the description matches any active search term keep it, and if it does not, allow the first credible scenario in stored order that matches any active search term to become the preview line instead of the non-matching description

This is **not** a ranking change.
It is one small search-aware preview selection improvement in the existing list seam.

## Product intent

This slice strengthens the core loop at:
- retrieve,
- inspect.

It should reduce friction between:
- "this result matched because of when-to-use context"
- and
- "the preview line actually shows me that relevant cue without opening detail view first."

## Why this slice is earned now

PromptManager already has a bounded shared preview seam that supports the main list.
Recent work already improved that seam through:
- active-search source-matched preview priority,
- match highlighting,
- body-lead fallback.

The remaining miss is small and local:
- active search can still show a generic description first,
- even when a stored scenario better explains why the prompt is relevant.

This slice improves the visible search explanation without touching ranking, filters, semantic search, or layout.

## Scope

### In scope
- one active-search-aware branch in the shared preview helper
- after the existing source-match check, keep the description when it matches any active search term
- otherwise, allow the first credible scenario in stored order that matches any active search term to become the preview line
- keep ordinary no-search preview order unchanged
- preserve existing active-search source-match priority unchanged
- add focused deterministic regression coverage for the new branch and unchanged current priorities

### Out of scope
- ranking changes
- filtering changes
- semantic search changes
- multi-line previews
- search-reason badges, chips, or extra metadata rows
- Draft Promote changes
- Quick Capture changes
- detail-view changes
- metadata/schema changes
- broad preview-policy redesign

## Recommended UX posture

Prefer one quiet active-search override only when it clearly improves visible match explanation.

Suggested v1:
- with no active search, keep the current preview order exactly as-is
- with active search:
  - matching credible source cue still wins first
  - matching description stays when it matches any active search term
  - otherwise, the first credible scenario in stored order that matches any active search term may override a non-matching description
- keep the result inside the existing single-line preview contract

Reason:
- it improves visible retrieval confidence
- it keeps the scope local to the current prompt-list preview seam
- it avoids turning preview selection into a scoring subsystem

## Data source

Use only existing prompt data already present in memory:
- `prompt.description`
- `prompt.scenarios`
- `prompt.source`
- active plain-text search terms already passed into the shared preview path

Do not add storage, caching, or new persistence in this slice.

## Likely implementation seam

### Shared preview helper
- `gui/prompt_preview.py`
  - extend `build_prompt_preview()` with one active-search scenario-priority branch
  - keep source-match priority first
  - keep no-search behavior unchanged
  - reuse existing flatten/truncate/credibility helpers

### Consumer
- `gui/prompt_list_model.py`
  - expected to benefit through existing `PreviewRole`
  - no new public role needed

### Tests
- `tests/test_prompt_list_model.py`
  - add focused coverage for scenario-priority during active search
  - protect unchanged no-search order
  - protect unchanged source-priority behavior
  - protect the case where a matching description still remains the right visible cue

## Happy-path scenario

1. A prompt has:
   - description: `Reusable incident handoff prompt.`
   - scenario: `Use after rollback review for release readiness decisions.`
2. User runs active plain-text search for `rollback review`.
3. The list row preview shows the scenario line instead of the generic description.
4. The operator understands immediately why the result is relevant.

That is enough for v1.

## Acceptance checks

1. With active plain-text search, after the existing source-match check, a first credible scenario in stored order can become the preview line when the description does not match any active search term.
2. With no active search, current preview order stays unchanged.
3. Existing active-search source-match priority stays unchanged.
4. A matching description remains the preview when it matches any active search term.
5. Search highlight continues to work on the chosen preview line.
6. No ranking, filtering, selection, layout, schema, or persistence behavior changes are introduced.
7. Focused regression coverage protects the new branch and the unchanged baseline.

## Suggested test set

A small deterministic set is enough:
- active-search matching scenario promoted over non-matching generic description
- no-search behavior unchanged
- existing source-match priority unchanged
- matching description still wins when it matches an active search term
- highlight roles still operate on the chosen preview text

## Rollback

Rollback should be one isolated patch:
- remove the active-search scenario-priority branch from `gui/prompt_preview.py`
- remove the focused regression tests added for it
- leave source priority, body fallback, and preview highlighting untouched

## Anti-goals

- do not redesign preview selection broadly
- do not change result ordering
- do not add search explanation UI beyond the existing preview line
- do not widen into scenario generation or refresh work
- do not mix Draft Promote into this slice
- do not create a generalized preview scoring framework

## Notes for implementation

- Keep the slice boring.
- Make the search-aware override explicit and narrow.
- Preserve current no-search behavior exactly.
- Reuse existing helper paths instead of inventing a new preview pipeline.
- If implementation pressure starts affecting ranking, layout, or additional consumers, the slice is drifting.
