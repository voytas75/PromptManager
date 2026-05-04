# PromptManager — Next tests Pyright slice plan v8

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the bounded `tests` Pyright slice selected after `tests/test_embedding_provider.py` and now closed by the shipped `tests/test_runtime_settings_service.py` cleanup.

## Goal

Keep the technical quality lane moving with one more bounded non-GUI `tests` slice before spending effort on the larger Qt/protected-access clusters.

## Current verified context

Confirmed by fresh live checks on 2026-05-04:
- `tests/test_web_search.py` was already closed in the prior slice note,
- `tests/test_embedding_provider.py` was already closed before this slice,
- `tests/test_runtime_settings_service.py` is now also closed,
- the file-level verification for the runtime-settings slice is green:
  - `uv run pyright tests/test_runtime_settings_service.py` → `0 errors`
  - `uv run pytest -q tests/test_runtime_settings_service.py` → `7 passed`
  - `uv run ruff check tests/test_runtime_settings_service.py` → `OK`
  - `uv run ruff format --check tests/test_runtime_settings_service.py` → `OK`
- fresh `uv run pyright tests --stats` now shows the tests backlog reduced further from **987 errors** to **979 errors**,
- the remaining backlog is still dominated by Qt/protected/private-usage files, but a few bounded non-GUI candidates remain.

Latest shipped checkpoint:
- implementation commit: `9116371`
- implementation commit message: `test: tighten runtime settings typing`

## Candidate summary at selection time

### `tests/test_runtime_settings_service.py`
- error count at pick time: **8**
- issue shape: object/optional subscript typing on a narrow assertion seam
- risk: low
- blast radius: test-only
- likely fix: introduce tighter typed locals or explicit `assert ... is not None` guards before subscripting returned objects

### Lower-priority candidates retained for later repick
- `tests/test_execution.py`
  - many mixed errors, including private-use and unknown monkeypatch/lambda typing
  - broader than needed for the next bounded slice
- `tests/test_share_controller.py`
  - non-GUI but noisier callback/lambda typing cluster
  - still a valid future candidate, but less mechanically narrow than the runtime-settings file
- Qt/protected-access files such as:
  - `tests/test_canonical_operator_path_parity.py`
  - `tests/test_draft_promote_dialog.py`
  - `tests/test_prompt_toolbar.py`
  - `tests/test_quick_capture_dialog.py`
  - these remain intentionally deferred because they are dominated by `reportPrivateUsage`

## Implemented slice

**Slice name:** `tests runtime settings service strict-typing cleanup`

### Candidate file
- `tests/test_runtime_settings_service.py`

### Why this slice was chosen
- still non-GUI and avoided the current protected/private-access wall,
- bounded to one file with only **8** live errors at pick time,
- issue shape was mechanical and assertion-local rather than architectural,
- lower-risk than jumping into mixed private-usage/callback-heavy files.

## Implemented approach

1. inspected the exact failing lines in `tests/test_runtime_settings_service.py`,
2. identified the narrow values Pyright still saw as `object | None`,
3. replaced raw subscripting with a typed local for `diagnostics["items"]`,
4. kept the change test-only and runtime-neutral,
5. reran the file-level verification loop and confirmed the slice was green.

## Scope boundary review

### In scope
- `tests/test_runtime_settings_service.py` only,
- clear the object/optional subscript typing errors,
- keep Ruff and formatting green,
- keep runtime behaviour unchanged,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production runtime-settings refactors,
- broader service typing cleanup outside this test file,
- GUI/protected-access slices.

## Files changed for the slice
- `tests/test_runtime_settings_service.py`
- this brief file

## Result

This slice is closed and shipped. The next work should be selected from a fresh repick after the post-v8 `tests --stats` snapshot, not from the pre-implementation candidate list alone.
