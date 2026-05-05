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

Net reduction across the continuation batch:
- from **958** -> **909**
- delta: **-49**

## Files changed and still uncommitted

Production:
- `gui/widgets/prompt_toolbar.py`
- `gui/dialogs/history.py`
- `gui/dialogs/quick_capture.py`
- `gui/dialogs/draft_promote.py`
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

Cross-check:
- `uv run pyright tests --stats` -> **909 errors**

## Recommended next slice on resume

Primary recommendation:
- `tests/test_quick_capture_dialog.py`
- current count: **8**
- shape: private-only GUI, likely needs public widget lookup / objectName seam

Alternative if we want a non-GUI mixed file next:
- `tests/test_execution.py`
- current count: do weryfikacji from fresh ranking, but currently still includes fixture typing + lambda typing + private helper usage
- shape: mixed, probably broader than the GUI seam slice

## Resume checklist

When resuming:
1. rerun `uv run pyright tests --stats`
2. confirm the backlog is still near **909**
3. choose one of:
   - GUI/private-only bounded seam: `uv run pyright tests/test_quick_capture_dialog.py`
   - broader mixed follow-up: `uv run pyright tests/test_execution.py`
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
