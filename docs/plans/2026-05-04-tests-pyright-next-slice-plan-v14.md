# PromptManager — Next tests Pyright slice plan v14

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the bounded `tests` Pyright slice selected after the implemented v13 cleanup of `tests/test_prompt_manager_reset.py` and now closed by the shipped `tests/test_settings_dialog_live_preview.py` cleanup.

## Goal

Keep the technical quality lane moving by clearing the next smallest private-only GUI test slice with the narrowest possible caller-side/public-surface seam.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v13 implementation commit `5173381`:
- `tests/test_prompt_manager_reset.py` is closed,
- `tests/test_settings_dialog_live_preview.py` is now also closed,
- file-level verification for the settings-dialog live-preview slice is green:
  - `uv run pyright tests/test_settings_dialog_live_preview.py` → `0 errors`
  - `uv run pytest -q tests/test_settings_dialog_live_preview.py` → `1 passed`
  - `uv run ruff check tests/test_settings_dialog_live_preview.py gui/settings_dialog.py` → `OK`
  - `uv run ruff format --check tests/test_settings_dialog_live_preview.py gui/settings_dialog.py` → `OK`
- fresh `uv run pyright tests --stats` now shows the tests backlog reduced further from **966 errors** to **963 errors**.

Latest shipped checkpoint:
- implementation commit: do zapisania po commicie bieżącej zmiany
- prior checkpoint before this pick: `5173381` (`test: tighten prompt manager reset typing`)

## Candidate summary at selection time

### `tests/test_settings_dialog_live_preview.py`
- error count at pick time: **3**
- issue shape: direct `reportPrivateUsage` on dialog internals:
  - `_inference_model_input`
  - `_workflow_groups`
  - `_model_input`
- risk: low-to-moderate
- blast radius: test plus a tiny production public-surface seam
- likely fix: route the test through public widget discovery (`findChild(...)`) backed by stable `objectName` values instead of direct protected access

## Implemented slice

**Slice name:** `tests settings dialog live preview public-widget access cleanup`

### Candidate file
- `tests/test_settings_dialog_live_preview.py`

### Why this slice was chosen
- next smallest visible private-only candidate,
- still narrow enough to solve with public widget lookup,
- avoided deeper production refactors or broader access-policy changes.

## Implemented approach

1. inspected the dialog structure and confirmed the preview already exposed `routingPreviewLabel` publicly,
2. added stable `objectName` values in `gui/settings_dialog.py` for:
   - `settingsFastModelInput`
   - `settingsInferenceModelInput`
   - `routingChoice_<workflow>_fast`
   - `routingChoice_<workflow>_inference`
3. rewrote the test to use `findChild(...)` with `QLineEdit` / `QRadioButton` instead of direct protected attribute access,
4. removed now-unused typing noise from the test,
5. reran the full file-level verification loop and confirmed the slice was green.

## Intended boundaries

### In scope
- `tests/test_settings_dialog_live_preview.py`,
- minimal supporting public widget naming in `gui/settings_dialog.py`,
- clear the three `reportPrivateUsage` errors,
- keep runtime behaviour unchanged,
- keep Ruff and formatting green,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- settings-dialog behavior changes,
- broader GUI test modernization,
- unrelated private/protected cleanup outside this slice.

## Files changed for the slice
- `tests/test_settings_dialog_live_preview.py`
- `gui/settings_dialog.py`
- this brief file

## Result

This slice is closed and shipped locally. The next work should be selected from a fresh post-v14 repick before continuing the tests Pyright lane.
