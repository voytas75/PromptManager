# Short search state continuity v1

Status: locally verified; remote delivery not requested
Owner: Wojtek / Prompt Manager Team
Canonical plan: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Verified hesitation

On the existing prompt-library toolbar, a one-character search request did not load results, but `PromptSearchController` still marked it active, changed the narrowing summary and locked manual sorting. The visible result set therefore appeared to belong to a request that had not run.

## Bounded contract

- Reject a non-empty, under-two-character explicit search request before changing active-search, summary, or sort state. Keep the last loaded result set and its cues unchanged.
- An accepted search still loads and shows its query. Clearing the field still restores the full prompt list and manual sort.
- Do not change ranking, persistence, CLI, provider calls, or other retrieval modes.

## Evidence

- RED: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_filter_panel.py::test_short_search_request_keeps_last_loaded_search_state` failed because sort was disabled after a one-character request with no load.
- GREEN: the focused panel/controller suite passed (`14 passed`); Ruff lint/format, narrow strict Pyright and `git diff --check` passed.
- Full provider-free gates passed: `958 passed, 1 skipped`, core coverage 80.97%, Ruff lint/format, full strict Pyright, `uv lock --check`, `git diff --check`. Remote CI remains unverified; no push is authorized in this slice.

## Next decision

After a verified local commit, choose the next user-visible asset-loop hesitation only on fresh evidence. Remote push/CI remain a separate delivery decision; do not multiply list-side guidance by default.
