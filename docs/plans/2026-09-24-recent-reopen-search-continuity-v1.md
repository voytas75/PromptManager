# Recent reopen across narrowed results v1

Status: behavior delivered as `e8773fee4f9e51369aca9905d66b4726fea3e64d`; exact-SHA Quality Gates [36031535329](https://github.com/voytas75/PromptManager/actions/runs/36031535329) succeeded; documentation checkpoint pending
Owner: PromptManager Team
Canonical near-term plan: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Reproduced hesitation

The `Recent` toolbar action previously supplied `RecentPromptsDialog` with `PromptListModel.prompts()`: the *visible*, currently filtered/search-result list. A recently modified catalog prompt absent from those results could not be reopened through `Recent`, even though the dialog promises recently touched prompts. Selecting an ID not in the model also cannot show detail: `WorkspaceHistoryController.select_prompt()` only scans visible rows.

## Bounded contract

- Supply recent candidates from the existing catalog, not the transient visible search/filter subset; preserve the dialog's deterministic sort and limit.
- If the selected recent record is already visible, select it without changing search or filters.
- If hidden, read the full catalog through the presenter before changing any view state. If that read fails or the record disappeared, leave the search/filter/selection unchanged and report it. On success clear the toolbar query, local category/tag/favorite/quality narrowing **and any pending presenter filter preferences** without intermediate reloads, restore manual sort and the full list from that already-read snapshot, persist the newly visible filter state, and select the record using the existing detail flow. No new panel, ranking, indexing, provider, or record mutation.
- Cancellation keeps the current view untouched. Initial Recent catalog-read failure uses the existing prompt-load error dialog; no silent fallback to a misleading subset.

## Evidence

- RED: three focused regressions for catalog-wide Recent candidates, hidden selection, and visible selection failed against the prior code (constructor did not accept catalog/reveal hooks; reveal operation did not exist).
- GREEN: `tests/test_recent_prompts.py`, `tests/test_prompt_filter_panel.py`, and `tests/test_retrieval_cues_parity.py`: 24 passed. Narrow Ruff lint/format and strict Pyright passed.
- Isolated offline real-GUI path in `tests/test_offline_gui_startup.py`: temporary deterministic-embedding catalog, narrowed visible search, Recent chooses hidden record, toolbar/list/detail recover to the full catalog. With recent dialog tests: 8 passed; narrow Ruff and Pyright green. No user database or provider was used.
- Two read-only reviews identified three blockers in successive candidates: uncaught initial Recent repository read, reset-before-fallible-read, and a pending presenter filter reapplied after clearing visible controls. Each was reproduced by a failing regression, then corrected. The handoff now preloads the catalog, fails without changing the old view, and clears pending filters before display. Final focused set: 31 passed; full local gates: 978 passed, 1 skipped, core coverage 81.66%; repo-wide Ruff lint/format, full strict Pyright, `uv lock --check`, and `git diff --check` passed. Behavior delivered as `e8773fee4f9e51369aca9905d66b4726fea3e64d`; exact-SHA Quality Gates [36031535329](https://github.com/voytas75/PromptManager/actions/runs/36031535329) succeeded.

## Next decision

The behavior commit `e8773fee4f9e51369aca9905d66b4726fea3e64d` reached `origin/master`; exact-SHA Quality Gates [36031535329](https://github.com/voytas75/PromptManager/actions/runs/36031535329) succeeded (Ruff, Pyright, pytest, clean tree). Documentation checkpoint remains to verify. Choose another asset-loop hesitation only on fresh operator evidence. Do not expand `Recent` into history or alter semantic ranking by default.
