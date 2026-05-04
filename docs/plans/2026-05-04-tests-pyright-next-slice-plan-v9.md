# PromptManager — Next tests Pyright slice plan v9

Status: proposed
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after the implemented v8 cleanup of `tests/test_runtime_settings_service.py`.

## Goal

Keep the technical quality lane moving with one more very small test-only Pyright slice that still avoids the large Qt/private-usage clusters.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v8 implementation commit `9116371`:
- `tests/test_runtime_settings_service.py` is closed,
- fresh `uv run pyright tests --stats` shows the tests backlog at **979 errors**,
- most remaining low-count files are still dominated by `reportPrivateUsage`,
- the smallest non-private candidate now visible is `tests/test_gui_diagnostics_status.py` with **2** live errors.

Latest shipped checkpoint before this pick:
- commit: `9116371`
- commit message: `test: tighten runtime settings typing`

## Candidate summary

### `tests/test_gui_diagnostics_status.py`
- error count: **2**
- issue shape: object/optional subscript typing on one assertion seam
- risk: very low
- blast radius: test-only
- likely fix: introduce a typed local or explicit non-`None` assertion before subscripting `diagnostics["items"]`

### Candidates checked but not selected
- `tests/test_prompt_chain_cli.py`
  - also **2** errors,
  - but both come from stub-vs-`PromptManager` parameter compatibility,
  - likely requires a broader protocol/cast decision than the diagnostics assertion seam
- `tests/test_prompt_manager_reset.py`
  - also **2** errors,
  - but both are `reportPrivateUsage` on `_create_calls`,
  - intentionally deferred with the other private-usage-heavy files
- `tests/test_settings_dialog_live_preview.py`
  - **3** errors,
  - still bounded, but less minimal than the diagnostics-status file

## Selected next slice

**Slice name:** `tests gui diagnostics status strict-typing cleanup`

### Candidate file
- `tests/test_gui_diagnostics_status.py`

### Why this slice was chosen
- smallest currently visible non-private candidate after v8,
- same mechanical error shape that was already resolved cleanly in the runtime-settings slice,
- likely solvable with a single typed-local assertion change,
- avoids escalating yet into CLI stub compatibility or private-usage policy decisions.

## Intended boundaries

### In scope
- `tests/test_gui_diagnostics_status.py` only,
- clear the two live object/optional subscript typing errors,
- keep runtime behaviour unchanged,
- keep Ruff and formatting green,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production GUI diagnostics refactors,
- broader settings diagnostics typing cleanup outside this test file,
- `reportPrivateUsage` clusters,
- prompt-chain CLI stub typing redesign.

## Proposed implementation approach

1. inspect the failing assertion around `diagnostics["items"]`,
2. introduce a typed local or explicit non-`None` guard before subscripting,
3. keep the change local to the test seam,
4. rerun:
   - `uv run pyright tests/test_gui_diagnostics_status.py`
   - `uv run pytest -q tests/test_gui_diagnostics_status.py`
   - `uv run ruff check tests/test_gui_diagnostics_status.py`
   - `uv run ruff format --check tests/test_gui_diagnostics_status.py`

## Likely files to change
- `tests/test_gui_diagnostics_status.py`
- this brief file

## Decision

This slice is the next recommended bounded test-only Pyright target after the implemented v8 runtime-settings cleanup.
