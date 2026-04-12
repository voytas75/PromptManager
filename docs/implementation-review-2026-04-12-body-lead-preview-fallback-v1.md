# PromptManager — Implementation Review

Date: 2026-04-12
Target: `Body-Lead Preview Fallback v1`
Expected sources:
- `docs/implementation-brief-2026-04-12-body-lead-preview-fallback-v1.md`
- `docs/delegation-brief-2026-04-12-body-lead-preview-fallback-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. It adds one final body-derived fallback inside the existing shared preview seam, preserves stronger preview priorities, and avoids widening into ranking, summarization, or adjacent Quick Capture behavior.

## What matches

### 1. The slice stays in the intended seam
The implementation remains local to:
- `gui/prompt_preview.py`
- `tests/test_prompt_list_model.py`
- `tests/test_draft_promote_dialog.py`

No search controller, list layout, Quick Capture, or persistence surfaces were changed.

### 2. The priority order stays aligned with the brief
`build_prompt_preview()` still prefers, in order:
- active-search matching credible source cue
- description
- scenarios
- credible source cue
- body lead fallback

That preserves the intended posture that body-derived preview is the last fallback, not a new primary path.

### 3. The fallback stays bounded and deterministic
The new body fallback:
- works only from existing `prompt.context`
- scans opening non-empty lines
- strips only simple markdown or label noise from those opening lines
- derives one short lead / sentence when credible
- reuses the shared flatten/truncate helpers
- refuses low-signal filler such as weak short placeholder body text

That is consistent with the brief's anti-drift posture.

### 4. The shared seam benefits the intended consumers
The implementation improves the shared preview contract that already feeds:
- list-style retrieval preview paths
- similar-match labels in `Draft Promote`

That matches the intended adjacent-core value without creating another surface.

### 5. Focused validation matches the slice
Validated locally with:
- `pytest -q tests/test_prompt_list_model.py tests/test_draft_promote_dialog.py`

Result:
- `24 passed`

## What is missing

Nothing material relative to the brief.

The patch includes the intended shared seam work and focused tests for both the main prompt list and the similar-match advisory reuse path.

## What drifted / widened

No meaningful scope drift is visible.

There is a slight heuristic expansion inside the body-preview helper because it strips simple markdown and label prefixes from opening lines before judging credibility.
That still stays inside the briefed seam and does not alter prompt storage, ranking, or UI structure.

## What is unverified

### 1. Manual UI pass
This review confirms code-path alignment and focused test coverage, but does not include a manual visual pass in the live list or `Draft Promote` dialog.

That is acceptable for this slice because the patch is local, deterministic, and covered through the shared preview seam tests.

## Recommended next action

Treat `Body-Lead Preview Fallback v1` as review-approved.

If you want to close the slice now, the next step is straightforward:
- keep the minimal docs touch,
- then commit and push.

## Sources reviewed

- `docs/implementation-brief-2026-04-12-body-lead-preview-fallback-v1.md`
- `docs/delegation-brief-2026-04-12-body-lead-preview-fallback-v1.md`
- `gui/prompt_preview.py`
- `tests/test_prompt_list_model.py`
- `tests/test_draft_promote_dialog.py`
- focused validation result (`24 passed`)
