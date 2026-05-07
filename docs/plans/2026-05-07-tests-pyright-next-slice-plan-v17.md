# PromptManager — Next tests Pyright slice plan v17

Status: implemented
Date: 2026-05-07
Updated: 2026-05-07
Source of truth: this note records the next bounded `tests` Pyright slice resumed after the last committed test-only cleanup.

## Goal

Continue with the smallest mechanical non-product slice visible in fresh live stats without overlapping the in-flight prompt-chain work.

## Current verified context

Confirmed by fresh live checks on 2026-05-07 before this slice:
- repo working tree was already dirty with unrelated prompt-chain files; this slice stayed isolated to `tests/test_share_controller.py`,
- fresh `uv run pyright tests --stats` showed the tests backlog at **808 errors**,
- `tests/test_share_controller.py` had **32** file-level errors,
- the file was test-only and the error cluster was mechanical:
  - untyped callback and monkeypatch lambdas,
  - untyped dummy indicator runner,
  - unused underscore-prefixed autouse fixture name.

## Candidate summary

### `tests/test_share_controller.py`
- error count: **32**
- issue shape: callback/lambda typing cleanup in a local test helper file
- risk: low
- blast radius: test-only
- likely fix:
  - type the dummy processing-indicator runner,
  - replace inline untyped lambdas with small typed helpers,
  - rename the autouse fixture to avoid the unused-function report.

## Selected next slice

**Slice name:** `tests share controller callback typing cleanup`

### Candidate file
- `tests/test_share_controller.py`

### Why this slice was chosen
- stays fully test-only,
- does not touch the currently modified prompt-chain runtime files,
- error shape is mechanical and reviewable,
- verification is cheap and file-local.

## Intended boundaries

### In scope
- `tests/test_share_controller.py`,
- clear the file-level Pyright errors,
- keep runtime behavior unchanged,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production sharing code,
- broader sharing test cleanup outside this file,
- commits or changes in the prompt-chain worktree slice.

## Implemented result

### Touched files
- `tests/test_share_controller.py`
- `docs/plans/2026-05-07-tests-pyright-next-slice-plan-v17.md`

### What changed
- typed `_DummyIndicator.run(...)` as a callable-based bool-returning helper,
- added typed `_noop_callback(...)` and `_always_true(...)` helpers to replace unknown callback lambdas,
- replaced the inline browser-open lambda with a typed local helper,
- renamed the autouse fixture from `_patch_processing_indicator` to `processing_indicator_fixture`.

### Verification
- `uv run pyright tests/test_share_controller.py` -> **0 errors**
- `uv run pytest -q tests/test_share_controller.py` -> **3 passed**
- `uv run ruff check --fix tests/test_share_controller.py` -> **1 import-order issue fixed; final state passed**
- `uv run ruff format --check tests/test_share_controller.py` -> **1 file already formatted**

### Runtime/product impact
- Runtime/product code stayed unchanged.

## Decision

This slice is implemented and verified. Next resume should re-scan fresh `uv run pyright tests --stats` output before picking another bounded file.
