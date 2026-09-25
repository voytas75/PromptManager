# PromptManager — search results hidden by filters (v1)

Status: RED → GREEN → full local gates verified; delivery state is confirmed separately by Git and exact-SHA CI
Date: 2026-09-25
Starting revision: `49be1d5b802695c668b440fe2eb10b34f360a17f` (`master`, clean and aligned with `origin/master`; exact-SHA Quality Gates [36134575476](https://github.com/voytas75/PromptManager/actions/runs/36134575476) succeeded).
Product authority: `docs/product-ssot.md`; near-term priority: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` §1 retrieval-to-action confidence. This is the bounded successor selected after `docs/plans/2026-09-25-project-review-followup-v2.md`, not a competing roadmap.

## Evidence and boundary

`gui/prompt_list_coordinator.py:56–85` distinguishes search hits, zero matches, and backend errors. `gui/prompt_list_presenter.py:380–397` filters a non-empty result list and clears detail if no row remains, but still emits the generic `Showing search results` status. The filter panel has a visible narrowing summary and `clear_narrowing()`; no ranking, persistence, CLI or new panel is needed. Existing focused list/filter/offline-GUI pack: 28 passed. No provider call, catalog mutation, user data or unrelated feature expansion.

## Ordered checkpoints

1. **RED — confirmed.** `tests/test_prompt_list_presenter.py::test_search_matches_hidden_by_favorites_filter_explain_empty_view` uses a non-favorite search hit with favorites-only filtering. The model is empty, detail is cleared, and no prompt selected; current status is incorrectly `Showing search results` instead of explaining that filters hide the match. First run exposed a test-fixture mistake (`model.prompts()` is a tuple, not a list), corrected the assertion, and the second run failed on the intended status mismatch (`1 failed`). Next: GREEN.
2. **GREEN — verified.** Changed only the presenter status branch for non-empty search hits hidden by filters; it now says `Search found matches, but active filters hide them — adjust filters to inspect.` No filter reset or search/ranking change. Targeted hidden-filter + ordinary-search pack: 2 passed; presenter/coordinator pack: 14 passed; touched-file Ruff/format and Pyright: 0 errors; `git diff --check` passed. Next: full verification/docs.
3. **Verify/docs — completed locally.** Focused search/list/filter/offline-GUI pack: 29 passed. Full provider-free suite: 1000 passed, 1 skipped; core coverage 81.66%. Whole-repo Ruff/format green (266 files), configured strict Pyright 252 files/0 errors, `uv lock --check` and `git diff --check` green. Added one `docs/CHANGELOG.md` Unreleased fix note. No remote CI for this successor.

## Progress

- RED: confirmed — targeted status mismatch after correcting tuple-vs-list test assumption.
- GREEN: verified — two targeted tests, 14 presenter/coordinator tests and touched-file checks passed.
- Verification: complete locally — 29 focused, 1000 full tests passed (1 skipped), core 81.66%; Ruff, configured Pyright and lock green.
- Delivery: see Git and exact-SHA CI for the current delivery state; the verification figures above are local evidence only.

## Stop rule and next decision

This one-seam status correction is locally verified. Do not continue with another wording-only search cue by inertia; select a successor only if a fresh operator-path test or observed session reveals a different hesitation or failure. The already delivered `Recent` handoff, ordinary search inspect hint and filter summary remain unchanged. Delivery status of this slice must be read from the current Git revision and its exact-SHA CI; local gates alone do not prove remote delivery.
