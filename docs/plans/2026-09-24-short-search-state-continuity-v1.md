# Short search state continuity v1

Status: delivered as `52f7cddca47f057d185ebe2b50d47abd2730c950` plus checkpoint `796f8bb1dd1028c5d473481dee7ac81ff3e00dad`; exact-SHA Quality Gates [36012781306](https://github.com/voytas75/PromptManager/actions/runs/36012781306) succeeded
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
- Full provider-free gates passed: `958 passed, 1 skipped`, core coverage 80.97%, Ruff lint/format, full strict Pyright, `uv lock --check`, `git diff --check`. Commits `52f7cdd` and `796f8bb` reached `origin/master`; exact-SHA Quality Gates run `36012781306` succeeded.

## Next decision

After delivered commit `52f7cdd` and verified checkpoint `796f8bb`, choose the next user-visible asset-loop hesitation only on fresh evidence; do not multiply list-side guidance by default.
