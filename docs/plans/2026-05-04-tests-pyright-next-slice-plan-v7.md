# PromptManager — Next tests Pyright slice plan v7

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after `tests/test_response_styles.py`.

## Goal

Keep the technical quality lane moving with another bounded non-GUI `tests` file before entering larger Qt/private-usage clusters.

## Current verified context

Confirmed by fresh live checks on 2026-05-04:
- `tests/test_response_styles.py` was already closed,
- the next useful non-GUI candidate was:
  - `tests/test_web_search.py` -> **7 errors**

File-level `uv run pyright tests/test_web_search.py` before implementation reported:
- four `pytest.approx(...)` usages flagged as partially unknown,
- one `monkeypatch.setattr(..., lambda seq: seq[1])` path flagged for unknown lambda parameter/return typing.

## Candidate summary

### `tests/test_web_search.py`
- error count: **7**
- issue shape: numeric assertion typing plus one typed monkeypatch helper
- risk: low
- blast radius: test-only
- likely fix: replace fragile `pytest.approx` typing with explicit float assertions and give the random-choice patch a typed helper function

## Selected next slice

**Slice name:** `tests web search strict-typing cleanup`

### Candidate file
- `tests/test_web_search.py`

### Why this slice was chosen
- still non-GUI and free of protected/private access debt
- bounded to one file with straightforward verification
- more mechanical than `runtime_settings_service` and lower-risk than Qt-heavy candidates
- useful because it removes a repeated strict-typing pattern around numeric assertions and monkeypatched callbacks

## Intended boundaries

### In scope
- `tests/test_web_search.py` only
- clear Pyright unknown-member / unknown-lambda issues
- keep Ruff TC and formatting green
- keep runtime behaviour unchanged
- verify with file-level pyright/pytest/ruff only

### Out of scope
- provider production-code refactors
- broader web-search typing cleanup outside this test file
- GUI/private-usage slices
- full `tests --stats` repick beyond this file

## Proposed implementation approach

1. inspect the exact `pytest.approx` and monkeypatch errors
2. replace the numeric comparisons with explicit `is not None` + bounded float assertions
3. replace the inline random-choice lambda with a typed local helper
4. gate typing-only imports behind `TYPE_CHECKING`
5. rerun:
   - `uv run pyright tests/test_web_search.py`
   - `uv run pytest -q tests/test_web_search.py`
   - `uv run ruff check tests/test_web_search.py`
   - `uv run ruff format --check tests/test_web_search.py`

## Likely files to change
- `tests/test_web_search.py`
- this brief file

## Decision

This slice is complete and implemented.

Implemented result on 2026-05-04:
- replaced four `pytest.approx(...)` assertions with explicit non-`None` float comparisons,
- replaced the inline `random.choice` monkeypatch lambda with a typed local helper,
- moved the typing-only `Sequence` import under `TYPE_CHECKING`,
- left runtime behaviour unchanged.

Verified:
- `uv run pyright tests/test_web_search.py` -> `0 errors`
- `uv run pytest -q tests/test_web_search.py` -> `18 passed`
- `uv run ruff check tests/test_web_search.py` -> `All checks passed!`
- `uv run ruff format --check tests/test_web_search.py` -> `1 file already formatted`

## Next repick guidance

We now have a cleaner checkpoint candidate.
Before opening a new product stage, finish the current technical batch by organizing the accumulated test-only slices and repo docs into a clean checkpoint.
