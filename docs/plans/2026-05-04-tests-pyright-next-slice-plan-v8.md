# PromptManager — Next tests Pyright slice plan v8

Status: proposed
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after `tests/test_embedding_provider.py`.

## Goal

Keep the technical quality lane moving with one more bounded non-GUI `tests` slice before spending effort on the larger Qt/protected-access clusters.

## Current verified context

Confirmed by fresh live checks on 2026-05-04:
- `tests/test_web_search.py` was already closed in the prior slice note,
- `tests/test_embedding_provider.py` is now also closed,
- fresh `uv run pyright tests --stats` dropped the tests backlog from **1001 errors** to **987 errors**,
- the remaining backlog is now dominated by Qt/protected/private-usage files, but there are still a few non-GUI candidates with mechanical typing issues.

Latest shipped checkpoint:
- commit: `2e20728`
- commit message: `test: tighten embedding provider typing`

## Candidate summary

### `tests/test_runtime_settings_service.py`
- error count: **8**
- issue shape: object/optional subscript typing on a narrow assertion seam
- risk: low
- blast radius: test-only
- likely fix: introduce tighter typed locals or explicit `assert ... is not None` guards before subscripting returned objects

### Lower-priority candidates for later repick
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

## Selected next slice

**Slice name:** `tests runtime settings service strict-typing cleanup`

### Candidate file
- `tests/test_runtime_settings_service.py`

### Why this slice was chosen
- still non-GUI and avoids the current protected/private-access wall,
- bounded to one file with only **8** live errors,
- issue shape looks mechanical and assertion-local rather than architectural,
- lower-risk than jumping into mixed private-usage/callback-heavy files.

## Intended boundaries

### In scope
- `tests/test_runtime_settings_service.py` only,
- clear the current object/optional subscript typing errors,
- keep Ruff and formatting green,
- keep runtime behaviour unchanged,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production runtime-settings refactors,
- broader service typing cleanup outside this test file,
- GUI/protected-access slices,
- another full `tests --stats` repick before this file is evaluated.

## Proposed implementation approach

1. inspect the exact failing lines in `tests/test_runtime_settings_service.py`,
2. identify the narrow values Pyright still sees as `object | None`,
3. replace raw subscripting with typed locals and explicit non-`None` guards,
4. keep any typing-only imports behind `TYPE_CHECKING` if needed,
5. rerun:
   - `uv run pyright tests/test_runtime_settings_service.py`
   - `uv run pytest -q tests/test_runtime_settings_service.py`
   - `uv run ruff check tests/test_runtime_settings_service.py`
   - `uv run ruff format --check tests/test_runtime_settings_service.py`

## Likely files to change
- `tests/test_runtime_settings_service.py`
- this brief file

## Decision

This slice is the next recommended bounded test-only Pyright target after `tests/test_embedding_provider.py`.
