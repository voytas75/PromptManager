# PromptManager — tests Pyright continuation status

Status: ready-for-resume
Date: 2026-05-04
Updated: 2026-05-04
Scope: bounded `tests/*` strict-Pyright reduction lane

## Current verified state

Fresh live baseline after the latest completed slice:
- `uv run pyright tests --stats` -> **909 errors**
- latest checked snapshot file: do weryfikacji (not persisted in this resume)

## Completed this run

Closed slices:
1. `tests/test_prompt_toolbar.py`
   - status: done
   - result: **-5**
   - pattern: public widget lookup via `objectName` in `gui/widgets/prompt_toolbar.py`

2. `tests/test_prompt_name_suggestion.py`
   - status: done
   - result: **-6**
   - pattern: mechanical test-stub typing cleanup, accessed availability probe for `PySide6`

3. `tests/test_prompt_version_history_dialog.py`
   - status: done
   - result: **-6**
   - pattern: public widget lookup via `objectName` in `gui/dialogs/history.py`

4. `tests/test_retrieval_cues_parity.py`
   - status: done
   - result: **-7**
   - pattern: typed `qt_app` fixture plus public widget lookup in:
     - `gui/dialogs/quick_capture.py`
     - `gui/dialogs/draft_promote.py`

5. `tests/test_prompt_model_roundtrip.py`
   - status: done
   - result: **-7**
   - pattern: public helper facades in `models/prompt_model.py` plus explicit float assertion instead of `pytest.approx(...)`

6. `tests/test_user_profile_preferences.py`
   - status: done
   - result: **-8**
   - pattern: typed `tmp_path: Path`, remove direct protected-method call by exercising public `suggest_prompts(...)`, keep personalization assertion through public result order

7. `tests/test_prompt_manager_mixins.py`
   - status: done
   - result: **-7**
   - pattern: local harness public wrapper/property seam over user-state initialisation, assertions moved off protected fields while keeping production code untouched

8. `tests/test_repository_branches.py`
   - status: done
   - result: **-8**
   - pattern: switch tests from private repository re-exports to public helpers in `core.repository.base`, replace `pytest.approx(...)` with explicit numeric assertions

9. `tests/test_quick_capture_dialog.py`
   - status: done
   - result: **-8**
   - pattern: public widget lookup via `objectName` plus a small public `build_draft()` facade in `gui/dialogs/quick_capture.py`

10. `tests/test_execution.py`
   - status: done
   - result: **-31**
   - pattern: swap private execution helpers for small public wrappers in `core/execution.py`, add typed completion-provider aliases, and annotate local token-counter test stubs

11. `tests/test_canonical_operator_path_parity.py`
   - status: done
   - result: **-15**
   - pattern: public widget lookup via existing `objectName` values in `PromptToolbar` and `PromptDetailWidget`; remove direct protected button access from the parity test

12. `tests/test_prompt_manager_storage.py`
   - status: done
   - result: **-25**
   - pattern: annotate `tmp_path: Path` on storage tests and keep `Path` under `TYPE_CHECKING` so repository/chroma path construction becomes type-safe without production changes

13. `tests/test_execute_context_history.py`
   - status: done
   - result: **-13**
   - pattern: replace private layout-state constants/helpers in the test with public facades and exported constants in `gui/layout_state.py`

14. `tests/test_factory_branches.py`
   - status: done
   - result: **-62**
   - pattern: replace private factory helper imports with public facades in `core/factory.py`, tighten `_make_settings(...)`, and replace monkeypatch lambdas with typed helpers

15. `tests/test_history_tracker.py`
   - status: done
   - result: **-26**
   - pattern: annotate `tmp_path: Path`, move `Path` under `TYPE_CHECKING`, and replace `pytest.approx(...)` with explicit numeric assertions

16. `tests/test_user_profile_preferences.py`
   - status: done
   - result: **-8**
   - pattern: annotate `tmp_path: Path` in repository/personalisation tests and add a public prompt-order wrapper in `core/prompt_manager/search.py` to avoid protected helper access

Net reduction across the continuation batch:
- from **958** -> **736**
- delta: **-222**

## Files changed and still uncommitted

Production:
- `gui/widgets/prompt_toolbar.py`
- `gui/dialogs/history.py`
- `gui/dialogs/quick_capture.py`
- `gui/dialogs/draft_promote.py`
- `gui/layout_state.py`
- `core/factory.py`
- `core/prompt_manager/search.py`
- `models/prompt_model.py`

Tests:
- `tests/test_prompt_toolbar.py`
- `tests/test_prompt_name_suggestion.py`
- `tests/test_prompt_version_history_dialog.py`
- `tests/test_retrieval_cues_parity.py`
- `tests/test_prompt_model_roundtrip.py`
- `tests/test_user_profile_preferences.py`
- `tests/test_prompt_manager_mixins.py`
- `tests/test_repository_branches.py`
- `tests/test_quick_capture_dialog.py`
- `tests/test_execution.py`
- `tests/test_canonical_operator_path_parity.py`
- `tests/test_prompt_manager_storage.py`
- `tests/test_execute_context_history.py`
- `tests/test_factory_branches.py`
- `tests/test_history_tracker.py`
- `tests/test_user_profile_preferences.py`

Plans/docs:
- `docs/plans/2026-05-04-tests-pyright-next-slice-plan-v15.md`
- `docs/plans/2026-05-04-tests-pyright-next-slice-plan-v16.md`
- `docs/plans/2026-05-04-tests-pyright-continuation-status.md`

## Verification completed in this resume

For `tests/test_user_profile_preferences.py`:
- `uv run pyright tests/test_user_profile_preferences.py` -> **0 errors**
- `uv run pytest -q tests/test_user_profile_preferences.py` -> **3 passed**
- `uv run ruff check tests/test_user_profile_preferences.py` -> **passed**
- `uv run ruff format --check tests/test_user_profile_preferences.py` -> **already formatted**

For `tests/test_prompt_manager_mixins.py`:
- `uv run pyright tests/test_prompt_manager_mixins.py` -> **0 errors**
- `uv run pytest -q tests/test_prompt_manager_mixins.py` -> **8 passed**
- `uv run ruff check tests/test_prompt_manager_mixins.py` -> **passed**
- `uv run ruff format --check tests/test_prompt_manager_mixins.py` -> **already formatted**

For `tests/test_repository_branches.py`:
- `uv run pyright tests/test_repository_branches.py` -> **0 errors**
- `uv run pytest -q tests/test_repository_branches.py` -> **32 passed**
- `uv run ruff check tests/test_repository_branches.py` -> **passed**
- `uv run ruff format --check tests/test_repository_branches.py` -> **already formatted**

For `tests/test_quick_capture_dialog.py`:
- `uv run pyright tests/test_quick_capture_dialog.py` -> **0 errors**
- `uv run pytest -q tests/test_quick_capture_dialog.py` -> **29 passed**
- `uv run ruff check tests/test_quick_capture_dialog.py gui/dialogs/quick_capture.py` -> **passed**
- `uv run ruff format --check tests/test_quick_capture_dialog.py gui/dialogs/quick_capture.py` -> **already formatted**

For `tests/test_execution.py`:
- `uv run pyright tests/test_execution.py` -> **0 errors**
- `uv run pytest -q tests/test_execution.py` -> **13 passed**
- `uv run ruff check tests/test_execution.py core/execution.py` -> **passed**
- `uv run ruff format --check tests/test_execution.py core/execution.py` -> **already formatted**

For `tests/test_canonical_operator_path_parity.py`:
- `uv run pyright tests/test_canonical_operator_path_parity.py` -> **0 errors**
- `uv run pytest -q tests/test_canonical_operator_path_parity.py` -> **1 passed**
- `uv run ruff check tests/test_canonical_operator_path_parity.py` -> **passed**
- `uv run ruff format --check tests/test_canonical_operator_path_parity.py` -> **already formatted**

For `tests/test_prompt_manager_storage.py`:
- `uv run pyright tests/test_prompt_manager_storage.py` -> **0 errors**
- `uv run pytest -q tests/test_prompt_manager_storage.py` -> **7 passed**
- `uv run ruff check tests/test_prompt_manager_storage.py` -> **passed**
- `uv run ruff format --check tests/test_prompt_manager_storage.py` -> **already formatted**

For `tests/test_execute_context_history.py`:
- `uv run pyright tests/test_execute_context_history.py` -> **0 errors**
- `uv run pytest -q tests/test_execute_context_history.py` -> **9 passed**
- `uv run ruff check tests/test_execute_context_history.py gui/layout_state.py` -> **passed**
- `uv run ruff format --check tests/test_execute_context_history.py gui/layout_state.py` -> **already formatted**

For `tests/test_factory_branches.py`:
- `uv run pyright tests/test_factory_branches.py` -> **0 errors**
- `uv run pytest -q tests/test_factory_branches.py` -> **10 passed**
- `uv run ruff check tests/test_factory_branches.py core/factory.py` -> **passed**
- `uv run ruff format --check tests/test_factory_branches.py core/factory.py` -> **already formatted**

For `tests/test_history_tracker.py`:
- `uv run pyright tests/test_history_tracker.py` -> **0 errors**
- `uv run pytest -q tests/test_history_tracker.py` -> **7 passed**
- `uv run ruff check tests/test_history_tracker.py` -> **passed**
- `uv run ruff format --check tests/test_history_tracker.py` -> **already formatted**

For `tests/test_user_profile_preferences.py`:
- `uv run pyright tests/test_user_profile_preferences.py` -> **0 errors**
- `uv run pytest -q tests/test_user_profile_preferences.py` -> **3 passed**
- `uv run ruff check tests/test_user_profile_preferences.py core/prompt_manager/search.py` -> **passed**
- `uv run ruff format --check tests/test_user_profile_preferences.py core/prompt_manager/search.py` -> **already formatted**

Cross-check:
- `uv run pyright tests --stats` -> **736 errors**

## Recommended next slice on resume

Primary recommendation:
- `tests/test_draft_promote_dialog.py`
- current count: **30**
- shape: public widget lookup / objectName seam across similarity summary, list widget, and title input; likely bounded but broader than the last parity slice

Alternative if we want another mechanical typing slice:
- `tests/test_user_profile_preferences.py`
- current count: **9**
- shape: `tmp_path: Path` typing plus one protected personalization helper to expose via public seam or wrapper

## Resume checklist

When resuming:
1. rerun `uv run pyright tests --stats`
2. confirm the backlog is still near **744**
3. choose one of:
   - broader GUI/private-only follow-up: `uv run pyright tests/test_draft_promote_dialog.py`
   - smaller typing/private seam: `uv run pyright tests/test_user_profile_preferences.py`
4. keep the slice bounded and verify with:
   - `uv run pyright <target-file>`
   - `uv run pytest -q <target-file>`
   - `uv run ruff check <target-file>`
   - `uv run ruff format --check <target-file>`

## Notes

- Current dirty tree is intentional and represents a coherent continuation batch.
- No commit/push done yet in this run.
- Repeated local warning remains benign but noisy:
  - `VIRTUAL_ENV=/home/voytas/.hermes/hermes-agent/venv does not match the project environment path .venv and will be ignored`
- The previous recommendations (`tests/test_user_profile_preferences.py`, `tests/test_prompt_manager_mixins.py`, `tests/test_repository_branches.py`) are now completed and should not be selected again.
