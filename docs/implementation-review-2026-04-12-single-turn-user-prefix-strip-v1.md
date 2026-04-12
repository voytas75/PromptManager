# PromptManager — Implementation Review

Date: 2026-04-12
Target: `Single-Turn User Prefix Strip v1`
Expected sources:
- `docs/implementation-brief-2026-04-12-single-turn-user-prefix-strip-v1.md`
- `docs/delegation-brief-2026-04-12-single-turn-user-prefix-strip-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. It adds one conservative `User:`-only cleanup rule inside the existing Quick Capture seam, protects transcript-like and non-`User:` cases, and avoids widening into broader role parsing.

## What matches

### 1. The slice stays in the intended seam
The implementation remains local to:
- `gui/dialogs/quick_capture.py`
- `tests/test_quick_capture_dialog.py`

No other prompt, storage, or UI surfaces were touched.

### 2. Only one bounded `User:` rule was added
The new helper strips exactly one outer `User:` wrapper for the two approved shapes:
- `User: ...`
- `User:\n...`

That matches the brief's narrow intended behavior.

### 3. Anti-drift guardrails are preserved
The implementation explicitly leaves these unchanged:
- bare `User:` with no real body
- transcript-like `User:` plus later role lines
- `Assistant:` prefixed input
- `System:` prefixed input

That is the right conservative posture.

### 4. Existing Quick Capture behavior remains intact
The helper is wired into `QuickCaptureDraft.to_prompt()` after the existing bounded cleanup helpers.
No schema, dialog fields, settings, or non-Quick-Capture behavior changed.

### 5. Focused validation matches the slice
Validated locally with:
- `.venv/bin/ruff check gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py`
- `.venv/bin/pyright gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py`
- `.venv/bin/pytest -q tests/test_quick_capture_dialog.py`

Results:
- Ruff: `All checks passed`
- Pyright: `0 errors, 0 warnings, 0 informations`
- Pytest: `28 passed`

## What is missing

Nothing material relative to the brief.

## What drifted / widened

No meaningful scope drift is visible.

The implementation did not widen into:
- `Assistant:` or `System:` stripping
- transcript parsing
- generic import cleanup
- broader Quick Capture refactoring

## What is unverified

### 1. Real operator frequency of this exact input shape
The review confirms the bounded implementation and focused tests, but it does not prove how often this `User:` wrapper occurs in live usage.

That is acceptable for this slice because the rule stays very small and low-risk.

## Recommended next action

Treat `Single-Turn User Prefix Strip v1` as review-approved.

If you want, the next step is straightforward:
- minimal doc update if desired,
- then commit and push.

## Sources reviewed

- `docs/implementation-brief-2026-04-12-single-turn-user-prefix-strip-v1.md`
- `docs/delegation-brief-2026-04-12-single-turn-user-prefix-strip-v1.md`
- `gui/dialogs/quick_capture.py`
- `tests/test_quick_capture_dialog.py`
- focused validation results listed above
