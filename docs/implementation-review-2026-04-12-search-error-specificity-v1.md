# PromptManager — Implementation Review

Date: 2026-04-12
Target: `Search Error Specificity v1`
Expected source: `docs/implementation-brief-2026-04-12-search-error-specificity-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. It stays inside the existing prompt-search failure seam, preserves one readable backend/Chroma detail line when available, and avoids widening into maintenance UX, search redesign, or error-framework refactoring.

## What matches

### 1. The change stays in the intended seam
The implementation remains local to:
- `core/prompt_manager/search.py`
- `tests/test_prompt_manager_branches.py`

That fits the brief's prompt-search error-handling posture.

### 2. Useful backend detail now survives the search failure path
Prompt search failures can now surface one sanitized backend detail line in the existing error path, for example a Chroma mismatch message, instead of collapsing every query failure into the same generic text.
That is the exact user-facing improvement the slice was supposed to provide.

### 3. No special-case lock-in was added
The implementation does not hardcode one single error class such as dimension mismatch.
Instead, it preserves readable backend detail more generally and falls back safely when the backend text is empty or unsuitable.
That matches the brief's anti-goal posture.

### 4. Sanitization stays proportionate
The helper keeps one readable line, avoids traceback noise, and truncates overly long backend text.
That is a good fit for the requested "show the real reason, but keep it readable" behavior.

### 5. Existing GUI popup flow stays intact
No presenter/dialog redesign was introduced.
The existing `Unable to search prompts` popup path simply receives a better underlying message from the same search seam.
That keeps the slice local and boring in the right way.

### 6. Focused regression coverage exists
Focused tests cover:
- a concrete backend query failure surfacing as sanitized user-facing detail
- traceback-only/noisy backend failures falling back to the generic search error

Validation passed:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_manager_branches.py -k 'surfaces_sanitized_backend_query_error or falls_back_for_unsuitable_backend_query_error'`
- result: `2 passed, 45 deselected`

## What is missing

Nothing material relative to the brief.

No presenter-level GUI test was added, but for this slice the search-seam unit coverage is proportionate and enough to verify the intended behavior.

## What drifted / widened

No meaningful scope drift is visible.

`docs/CHANGELOG.md` was updated minimally, which is consistent with repo posture and not product-scope widening.

## What is unverified

### 1. Live popup readability in a manual GUI pass
This review did not include a manual GUI pass for message wrapping or line-break feel in the dialog itself.

### 2. Broader backend error variety
The focused tests prove the bounded error-shaping path, but this review did not manually exercise a larger catalog of real backend failures beyond the representative test cases.

## Recommended next action

Treat `Search Error Specificity v1` as delivered.

Do not widen it into repair buttons, maintenance hints, or a broader error taxonomy unless a new separate slice is explicitly chosen.

## Sources reviewed

- `docs/implementation-brief-2026-04-12-search-error-specificity-v1.md`
- `core/prompt_manager/search.py`
- `tests/test_prompt_manager_branches.py`
- focused test result: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_manager_branches.py -k 'surfaces_sanitized_backend_query_error or falls_back_for_unsuitable_backend_query_error'` → `2 passed, 45 deselected`
