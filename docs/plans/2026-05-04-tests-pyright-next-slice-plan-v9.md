# PromptManager — Next tests Pyright slice plan v9

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the bounded `tests` Pyright slice selected after the implemented v8 cleanup of `tests/test_runtime_settings_service.py` and now closed by the shipped `tests/test_gui_diagnostics_status.py` cleanup.

## Goal

Keep the technical quality lane moving with one more very small test-only Pyright slice that still avoids the large Qt/private-usage clusters.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v8 implementation commit `9116371`:
- `tests/test_runtime_settings_service.py` is closed,
- `tests/test_gui_diagnostics_status.py` is now also closed,
- file-level verification for the GUI diagnostics status slice is green:
  - `uv run pyright tests/test_gui_diagnostics_status.py` → `0 errors`
  - `uv run pytest -q tests/test_gui_diagnostics_status.py` → `4 passed`
  - `uv run ruff check tests/test_gui_diagnostics_status.py` → `OK`
  - `uv run ruff format --check tests/test_gui_diagnostics_status.py` → `OK`
- fresh `uv run pyright tests --stats` now shows the tests backlog reduced further from **979 errors** to **977 errors**,
- most remaining low-count files are still dominated by `reportPrivateUsage`, but a few bounded non-GUI candidates remain.

Latest shipped checkpoint:
- implementation commit: do zapisania po commicie bieżącej zmiany
- prior checkpoint before this pick: `9116371` (`test: tighten runtime settings typing`)

## Candidate summary at selection time

### `tests/test_gui_diagnostics_status.py`
- error count at pick time: **2**
- issue shape: object/optional subscript typing on one assertion seam
- risk: very low
- blast radius: test-only
- likely fix: introduce a typed local or explicit non-`None` assertion before subscripting `diagnostics["items"]`

### Candidates checked but not selected
- `tests/test_prompt_chain_cli.py`
  - also **2** errors,
  - but both came from stub-vs-`PromptManager` parameter compatibility,
  - likely requires a broader protocol/cast decision than the diagnostics assertion seam
- `tests/test_prompt_manager_reset.py`
  - also **2** errors,
  - but both are `reportPrivateUsage` on `_create_calls`,
  - intentionally deferred with the other private-usage-heavy files
- `tests/test_settings_dialog_live_preview.py`
  - **3** errors,
  - still bounded, but less minimal than the diagnostics-status file

## Implemented slice

**Slice name:** `tests gui diagnostics status strict-typing cleanup`

### Candidate file
- `tests/test_gui_diagnostics_status.py`

### Why this slice was chosen
- smallest currently visible non-private candidate after v8,
- same mechanical error shape that was already resolved cleanly in the runtime-settings slice,
- solvable with one typed-local assertion change,
- avoided escalating into CLI stub compatibility or private-usage policy decisions.

## Implemented approach

1. inspected the failing assertion around `diagnostics["items"]`,
2. added typing-only imports behind `TYPE_CHECKING`,
3. introduced a narrow typed local:
   - `items = cast("list[Mapping[str, object]]", diagnostics["items"])`
4. kept the change local to the test seam and runtime-neutral,
5. reran the full file-level verification loop and confirmed the slice was green.

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

## Files changed for the slice
- `tests/test_gui_diagnostics_status.py`
- this brief file

## Result

This slice is closed and shipped locally. The next work should be selected from a fresh post-v9 repick before continuing the tests Pyright lane.
