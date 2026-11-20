# Retrieval → inspect → reuse: bounded execution

Status: done — delivered on `origin/master`, exact-SHA CI passed
Baseline: `3bfe623afef65a18d23d0bc7744f08ef22af2298`
Delivered product commit: `897260fea994cc8242005bca40cb253f21c93ecb`
CI: [Quality Gates 44378113049](https://github.com/voytas75/PromptManager/actions/runs/44378113049) — success; 1749 passed, 1 skipped; core coverage 82.25% (2026-11-18).
Delivery verification: local HEAD matched live `origin/master`; exact-SHA CI status and logs checked on 2026-11-20. This documentation correction is a separate follow-up and is not covered by that product CI run.
Evidence boundary: historical local checkpoints below retain their original no-commit/push scope. Subsequent delivery is recorded here; native GUI acceptance and a fresh full local pytest/coverage run are not claimed.
Owner: Wojtek
Product authority: `docs/product-ssot.md`
Near-term direction: `2026-05-10-product-direction-ssot-next-cycle.md`

## Approved scope
Implement the three candidates from the Astra discovery sequentially. Codex model: `gpt-6.2-astra`; Hermes independently verifies results. Update this ledger after each work item and before the next. No commit/push, dependency/config/security changes, provider-backed application calls, real catalog access, ranking redesign or unrelated refactoring. Codex inference is authorized by the requested Codex continuation.

## Sequence and acceptance
- [x] S1 Unicode highlighting: reproduce expanding casefold indices; map normalized matches back to original spans. Cover ß, expansion before a match, overlapping terms and ordinary ASCII. Target `gui/prompt_list_model.py`, `tests/test_prompt_list_model.py`.
- [x] S2 Filter selection continuity: first reproduce existing behavior; preserve selected UUID only if still visible. If excluded, clear both selection and detail state. Test retained/excluded/empty results and existing sort behavior. Target `gui/prompt_list_presenter.py`, `tests/test_prompt_list_presenter.py` and nearest existing presenter seams only if necessary.
- [x] S3 Copy availability: use existing stored-body predicate in context menu. Cover None/empty/whitespace body with a description and a valid body; preserve workspace fallback. Target `gui/prompt_actions_controller.py`, `tests/test_prompt_actions_controller.py`.
- [x] Final acceptance: affected/adjacent tests, repository Ruff check/format check, changed-file Pyright, diff/scope review; changelog and status pointer. Native GUI acceptance not performed.

## Execution rules
For each slice: tests first → attributable RED → minimal implementation → GREEN → immediate ledger update → independent Hermes verification. If a candidate already passes, record covered behavior instead of forcing production changes. Use existing project environment; no installs. Test harness must block network, suppress dotenv, clear provider credentials, isolate HOME/XDG and redirect temporary test state outside the real catalog. Retained receipts: `/home/voytas/.hermes/artifacts/promptmanager-retrieval-reuse/`.

## Initial checkpoint
Plan created; clean baseline confirmed. Discovery reproduced S1 `IndexError` by executing the unchanged method in isolation; this is method-level evidence, not GUI acceptance. S1 is next.

### S1 — implemented, 2026-11-18

- Common runner from the repository root: `env -i PATH=/usr/bin:/bin LANG=C.UTF-8 /home/voytas/projects/PromptManager/.venv/bin/python -IB /home/voytas/.hermes/artifacts/promptmanager-retrieval-reuse/run-01/guarded_pytest.py`, followed by the selectors below. Fresh HOME/XDG/temp/catalog paths per run, offscreen Qt, empty provider environment, dotenv disabled, local LiteLLM cost map, socket/DNS and protected-file audit guard.
- Harness preflight (not RED): Chroma's import-time settings reader attempted to open `.env`; the guard denied it before read (1 collection error, cleanup also errored). The runner now suppresses the independent `dotenv_values` reader and uses an unambiguous disposable XDG directory name.
- RED: `tests/test_prompt_list_model.py::test_title_highlights_use_displayed_character_positions --tb=short` → **4 failed, 1 passed in 2.49s**, exit 1, guard blocks `{}`. Three `IndexError`s (whole expansion, overlapping terms, partial expansion); expansion before a match returned `(6, 6)` instead of `(5, 6)`. ASCII passed.
- Folded match endpoints now map to displayed character positions before checking overlap or returning spans. GREEN: `tests/test_prompt_list_model.py` → **32 passed in 1.70s**, exit 0, guard blocks `{}`. Independent verification pending.

### S2 — implemented, 2026-11-18

- Traced before implementation: `PromptSearchController.filters_changed` calls `refresh_filtered_view`; the presenter selects via `MainWindow._select_prompt → WorkspaceHistoryController.select_prompt → QListView.setCurrentIndex`. The list's `selectionChanged` handler reads the presenter's current index and updates the main and template details, preview and execution selection. Model reset alone does not send that selection notification.
- RED: `tests/test_prompt_list_presenter.py::test_filter_refresh_synchronizes_selection_and_details --tb=short` → **3 failed in 1.80s**, exit 1, guard blocks `{}`. Using the real selection handler and both detail widgets: retained UUID lost; excluded UUID remained in main detail with another result visible; empty list cleared main detail but retained template detail. Sort retained the UUID at its new row before all three failures.
- Capture the selected row before reset; restore its UUID through the existing callback only if visible. For an excluded selection, clear current index then selection before reset so the existing handler clears dependent state. No change to sort. GREEN: `tests/test_prompt_list_presenter.py` → **8 passed in 1.67s**, exit 0, guard blocks `{}`. Independent verification pending.

### S3 — implemented, 2026-11-18

- RED: `tests/test_prompt_actions_controller.py -k 'copy_requires_stored_body or seeds_text_without_running' --tb=short` → **3 failed, 4 passed, 10 deselected in 1.67s**, exit 1, guard blocks `{}`. With a description, Copy Prompt was enabled for None, empty and whitespace bodies. Valid-body copy availability and all three non-executing workspace cases already passed.
- The menu and clipboard handler now use the same `_copyable_prompt_body` check. Workspace payload selection is unchanged; None and empty bodies still fall back to the description. GREEN: `tests/test_prompt_actions_controller.py` → **17 passed in 1.72s**, exit 0, guard blocks `{}`. Independent verification pending.

### Completion — 2026-11-18

- Combined affected + nearby smoke: the three full affected test modules, `tests/test_prompt_list_coordinator.py`, `tests/test_prompt_filter_panel.py`, and `tests/test_prompt_detail_widget.py::{test_prompt_detail_widget_disables_copy_without_a_prompt_body,test_prompt_detail_widget_exposes_bounded_quick_reuse_actions}` (two node IDs). Guarded runner: **83 passed in 2.02s**; final rerun after formatting: **83 passed in 2.00s**, exit 0, no skips or guard blocks.
- For the six Python files named in S1–S3: `.venv/bin/ruff check --no-cache` passed; `.venv/bin/ruff format --check --no-cache` initially requested one line break in the S2 test and then reported **6 files already formatted**. Bundled `.venv/bin/pyright` with version lookup disabled: **0 errors, 0 warnings, 0 informations**, exit 0.
- Reviewed diff and scope; `git diff --check` passed. Updated the unreleased changelog and the single status pointer. HEAD remains at the baseline; the preexisting untracked ledger remains untracked. Outside the repository, only the execution harness was added under the approved run-01 root; disposable test directories were removed.
- Native GUI acceptance was not performed. Full-project pytest/coverage was not rerun; bounded acceptance is recorded below.

### Independent Hermes acceptance — 2026-11-18
- S1 independently rerun after its implementation: **32 passed**, guard blocks `{}`; source diff inspected before S2 completion.
- Final seven-module affected/adjacent run: `tests/test_prompt_list_model.py tests/test_prompt_list_presenter.py tests/test_prompt_actions_controller.py tests/test_prompt_list_coordinator.py tests/test_prompt_filter_panel.py tests/test_main_window_bridges.py tests/test_template_preview_widget.py` using `.venv/bin/python -I -B` and the guarded runner above: **93 passed in 1.84s**, exit 0, guard blocks `{}`.
- Additional complete detail-widget module: **46 passed in 1.98s**, exit 0, guard blocks `{}`. These are separate execution receipts, not a full-suite claim.
- Repository `.venv/bin/ruff check .` passed; `.venv/bin/ruff format --check .`: **291 files already formatted**. Explicit Pyright over all six changed Python files: **0 errors / warnings / informations**. Its newer-version notice was not acted on. `git diff --check` passed.
- Parent inspected full scoped production/test/docs diff and native Codex RED/GREEN command receipts. Postprocessing receipt inspection initially raised a TypeError on a null aggregatedOutput after printing all six requested RED/GREEN receipts; this did not affect product tests or Codex completion.
- Scope: three production files, three test files, changelog/status and this ledger only. No commit/push, catalog changes or application provider calls. All approved work items closed; no successor added automatically.
