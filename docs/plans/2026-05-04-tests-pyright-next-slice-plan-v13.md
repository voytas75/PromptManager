# PromptManager — Next tests Pyright slice plan v13

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the first bounded `tests` Pyright slice selected from the remaining private-usage-only candidates after the implemented v12 cleanup of `tests/test_main_window_bridges.py` and now closed by the shipped `tests/test_prompt_manager_reset.py` cleanup.

## Goal

Keep the technical quality lane moving by clearing the smallest remaining private-usage-only test slice with a caller-side fix before considering broader GUI protected-access work.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v12 implementation commit `029f014`:
- `tests/test_main_window_bridges.py` is closed,
- `tests/test_prompt_manager_reset.py` is now also closed,
- file-level verification for the reset slice is green:
  - `uv run pyright tests/test_prompt_manager_reset.py` → `0 errors`
  - `uv run pytest -q tests/test_prompt_manager_reset.py` → `6 passed`
  - `uv run ruff check tests/test_prompt_manager_reset.py` → `OK`
  - `uv run ruff format --check tests/test_prompt_manager_reset.py` → `OK`
- fresh `uv run pyright tests --stats` now shows the tests backlog reduced further from **968 errors** to **966 errors**,
- `tests/test_settings_dialog_live_preview.py` remains the next smallest visible private-only candidate with **3** errors.

Latest shipped checkpoint:
- implementation commit: do zapisania po commicie bieżącej zmiany
- prior checkpoint before this pick: `029f014` (`test: tighten main window bridges typing`)

## Candidate summary at selection time

### `tests/test_prompt_manager_reset.py`
- error count at pick time: **2**
- issue shape: direct test access to protected stub counters (`_create_calls`)
- risk: very low
- blast radius: test-only
- likely fix: rename the stub counter to a public attribute such as `create_calls` and update the internal increment plus assertions accordingly

### Candidates checked but not selected
- `tests/test_settings_dialog_live_preview.py`
  - **3** errors,
  - all `reportPrivateUsage` on real dialog internals,
  - higher-risk than the stub-counter caller-side rename

## Implemented slice

**Slice name:** `tests prompt manager reset private-counter cleanup`

### Candidate file
- `tests/test_prompt_manager_reset.py`

### Why this slice was chosen
- smallest remaining private-only candidate,
- the private access was against local test stubs, not production internals,
- solvable by a pure caller-side rename with no runtime behavior change.

## Implemented approach

1. renamed the stub counter field from `_create_calls` to `create_calls`,
2. updated the increment sites in both stub clients,
3. updated the two assertions to the public counter name,
4. reran the full file-level verification loop and confirmed the slice was green.

## Intended boundaries

### In scope
- `tests/test_prompt_manager_reset.py` only,
- clear the two `reportPrivateUsage` errors on `_create_calls`,
- keep runtime behaviour unchanged,
- keep Ruff and formatting green,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production reset logic changes,
- GUI private-access slices,
- broader protected/private policy decisions beyond this stub seam.

## Files changed for the slice
- `tests/test_prompt_manager_reset.py`
- this brief file

## Result

This slice is closed and shipped locally. The next work should be selected from a fresh post-v13 repick before continuing the tests Pyright lane.
