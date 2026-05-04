# PromptManager — Next tests Pyright slice plan v12

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after the implemented v11 cleanup of `tests/test_template_preview_widget.py` and now closed by the shipped `tests/test_main_window_bridges.py` cleanup.

## Goal

Keep the technical quality lane moving by preferring the smallest remaining non-private test slice before entering direct `reportPrivateUsage` cleanup.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v11 implementation commit `8116e32`:
- `tests/test_template_preview_widget.py` is closed,
- `tests/test_main_window_bridges.py` is now also closed,
- file-level verification for the main-window-bridges slice is green:
  - `uv run pyright tests/test_main_window_bridges.py` → `0 errors`
  - `uv run pytest -q tests/test_main_window_bridges.py` → `2 passed`
  - `uv run ruff check tests/test_main_window_bridges.py` → `OK`
  - `uv run ruff format --check tests/test_main_window_bridges.py` → `OK`
- fresh `uv run pyright tests --stats` now shows the tests backlog reduced further from **972 errors** to **968 errors**,
- the smallest remaining files are now private-only candidates such as:
  - `tests/test_prompt_manager_reset.py` → **2** errors (`reportPrivateUsage` only)
  - `tests/test_settings_dialog_live_preview.py` → **3** errors (`reportPrivateUsage` only)

Latest shipped checkpoint:
- implementation commit: do zapisania po commicie bieżącej zmiany
- prior checkpoint before this pick: `8116e32` (`test: tighten template preview widget typing`)

## Candidate summary at selection time

### `tests/test_main_window_bridges.py`
- error count at pick time: **4**
- issue shape: partially unknown callback/member types on `bridge.delete_current_prompt` and `bridge.close_application`
- risk: low
- blast radius: test-only
- likely fix: add explicit callable typing at the captured callback seam, and replace the no-op fallback lambda with a tiny typed helper if needed

### Candidates checked but not selected
- `tests/test_prompt_manager_reset.py`
  - fewer raw errors (**2**), but both are `reportPrivateUsage`,
  - intentionally deferred because a non-private candidate still existed
- `tests/test_settings_dialog_live_preview.py`
  - **3** errors, all `reportPrivateUsage` on dialog internals,
  - also deferred until the non-private lane was exhausted

## Implemented slice

**Slice name:** `tests main window bridges strict-typing cleanup`

### Candidate file
- `tests/test_main_window_bridges.py`

### Why this slice was chosen
- smallest visible non-private candidate,
- narrow error cluster on callback typing only,
- solvable without touching production code or opening a private-access policy decision.

## Implemented approach

1. inspected the captured callback assignments from `PromptActionsBridge`,
2. replaced the inline no-op fallback lambda with a tiny typed helper,
3. cast the bridge-bound callbacks through `Any` into explicit `Callable[[], None]` shapes at capture time,
4. moved the `Callable` import into the `TYPE_CHECKING` block to satisfy Ruff TC rules,
5. ran `ruff check --fix` once to normalize import ordering,
6. reran the full file-level verification loop and confirmed the slice was green.

## Intended boundaries

### In scope
- `tests/test_main_window_bridges.py` only,
- clear the four partially-unknown callback/member typing errors,
- keep runtime behaviour unchanged,
- keep Ruff and formatting green,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production bridge redesign,
- private-usage slices,
- broader GUI handler typing cleanup outside this test file.

## Files changed for the slice
- `tests/test_main_window_bridges.py`
- this brief file

## Result

This slice is closed and shipped locally. The next work should be selected from a fresh post-v12 repick before continuing the tests Pyright lane.
