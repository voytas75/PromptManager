# PromptManager — Implementation Review

Date: 2026-04-11
Target: `Prompt Label Strip v1`
Expected source: `docs/implementation-brief-2026-04-11-prompt-label-strip-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. It adds one narrow Quick Capture normalization step, keeps the cleanup limited to a tiny allowlist and top-of-body seam, and leaves ambiguous or transcript-like input untouched.

## What matches

### 1. The change stays inside the intended seam
The implementation remains local to:
- `gui/dialogs/quick_capture.py`
- `tests/test_quick_capture_dialog.py`

That matches the brief's Quick Capture-only posture.

### 2. The label-strip logic is intentionally narrow
The helper strips at most one outer label and only for the allowed forms:
- `Prompt:`
- `User prompt:`
- `System prompt:`

It only activates for the two bounded shapes described in the brief:
- `Label: prompt text...`
- `Label:\nprompt text...`

### 3. Ambiguous and transcript-like cases stay unchanged
The implementation explicitly keeps these cases untouched:
- bare labels such as `Prompt:` with no body
- ordinary later mentions such as `... prompt: ...` in the middle of text
- transcript-like follow-on content starting with role markers such as `User:` or `Assistant:`

That matches the false-negative-over-destructive-cleanup posture.

### 4. Existing flow and storage stay intact
The change is applied locally during `QuickCaptureDraft.to_prompt()` after the existing fence-unwrap step.
No schema, UI surface, or broader import/transcript behavior was introduced.

### 5. Focused validation passed
Focused tests cover:
- stripped inline allowed label
- stripped multiline allowed label
- unchanged ambiguous / incomplete / transcript-like input
- the existing quick-capture seam remaining healthy

Validation passed:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_quick_capture_dialog.py`
- result: `11 passed`

## What is missing

Nothing material relative to the brief.

No broader role parsing was added, which is correct for this slice and should remain out of scope.

## What drifted / widened

No meaningful scope drift is visible.

The implementation remains small and local. It does not turn Quick Capture into a generic parser or cleanup framework.

## What is unverified

### 1. Live operator frequency
This review did not measure how often real captured prompts arrive with these exact wrapper labels in day-to-day use.

### 2. Longer mixed-format edge cases
The bounded tests prove the intended happy path and key guarded paths, but this review did not explore broader real-world mixtures of prose, labels, and formatting beyond the brief.

## Recommended next action

Treat `Prompt Label Strip v1` as delivered.

Do not widen it into transcript parsing.
If follow-up is needed later, it should be another separate tiny cleanup slice with equally tight guardrails.

## Sources reviewed

- `docs/implementation-brief-2026-04-11-prompt-label-strip-v1.md`
- `gui/dialogs/quick_capture.py`
- `tests/test_quick_capture_dialog.py`
- focused test result: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_quick_capture_dialog.py` → `11 passed`
