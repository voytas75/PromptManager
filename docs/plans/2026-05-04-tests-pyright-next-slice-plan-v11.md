# PromptManager — Next tests Pyright slice plan v11

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after the implemented v10 cleanup of `tests/test_prompt_chain_cli.py` and now closed by the shipped `tests/test_template_preview_widget.py` cleanup.

## Goal

Keep the technical quality lane moving with another ultra-small test-only Pyright slice while still avoiding direct `reportPrivateUsage` clusters.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v10 implementation commit `286bdf8`:
- `tests/test_prompt_chain_cli.py` is closed,
- `tests/test_template_preview_widget.py` is now also closed,
- file-level verification for the template-preview slice is green:
  - `uv run pyright tests/test_template_preview_widget.py` → `0 errors`
  - `uv run pytest -q tests/test_template_preview_widget.py` → `10 passed`
  - `uv run ruff check tests/test_template_preview_widget.py` → `OK`
  - `uv run ruff format --check tests/test_template_preview_widget.py` → `OK`
- fresh `uv run pyright tests --stats` now shows the tests backlog reduced further from **975 errors** to **972 errors**,
- `tests/test_settings_dialog_live_preview.py` still has **3** errors, but they are all direct `reportPrivateUsage` on dialog internals,
- the remaining best candidates should be re-picked from the fresh post-v11 surface.

Latest shipped checkpoint:
- implementation commit: do zapisania po commicie bieżącej zmiany
- prior checkpoint before this pick: `286bdf8` (`test: tighten prompt chain cli typing`)

## Candidate summary at selection time

### `tests/test_template_preview_widget.py`
- error count at pick time: **3**
- issue shape: unused import guard plus partially-unknown lambda params in a test stub
- risk: low
- blast radius: test-only
- likely fix: replace the guard import with an accessed availability check, and swap lambda placeholders for tiny typed helper callables

### Candidates checked but not selected
- `tests/test_settings_dialog_live_preview.py`
  - also **3** errors,
  - but all are direct `reportPrivateUsage` against `_inference_model_input`, `_workflow_groups`, and `_model_input`,
  - intentionally deferred behind lower-risk mechanical test slices
- `tests/test_prompt_manager_reset.py`
  - **2** errors,
  - but both are `reportPrivateUsage`,
  - still lower priority than the template-preview mechanical file

## Implemented slice

**Slice name:** `tests template preview widget strict-typing cleanup`

### Candidate file
- `tests/test_template_preview_widget.py`

### Why this slice was chosen
- same raw size as the next GUI candidate, but with a lower-risk non-private error shape,
- solvable entirely inside test stubs,
- did not require a production API seam or protected-access decision.

## Implemented approach

1. replaced the `import PySide6` guard with `importlib.util.find_spec("PySide6")`,
2. inspected the lambda-based `SimpleNamespace` stubs around the state-store seam,
3. replaced inline lambdas with tiny typed helper functions,
4. ran Ruff formatting once after the typing patch,
5. reran the full file-level verification loop and confirmed the slice was green.

## Intended boundaries

### In scope
- `tests/test_template_preview_widget.py` only,
- clear the one unused-import error and two unknown-lambda errors,
- keep runtime behaviour unchanged,
- keep Ruff and formatting green,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- template preview production refactors,
- GUI protected-access rewrites,
- broader Qt test cleanup outside this file.

## Files changed for the slice
- `tests/test_template_preview_widget.py`
- this brief file

## Result

This slice is closed and shipped locally. The next work should be selected from a fresh post-v11 repick before continuing the tests Pyright lane.
