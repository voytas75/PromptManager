# PromptManager — Next tests Pyright slice plan v16

Status: proposed
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after the implemented v15 toolbar cleanup.

## Goal

Continue with the smallest mechanical non-GUI test slice visible in fresh live stats.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v15 toolbar slice:
- `tests/test_prompt_toolbar.py` is green,
- fresh `uv run pyright tests --stats` shows the tests backlog at **958 errors**,
- the smallest visible candidate is `tests/test_prompt_name_suggestion.py` with **6** errors,
- this file is **mixed**, but still mechanical and test-local:
  - one unused import guard for `PySide6`,
  - untyped `*args, **kwargs` in a local test stub,
  - one partially unknown `super().__init__()` path in the local stub.

Latest shipped checkpoint before this pick:
- base commit before v15 plan: `5d95792`
- v15 work is currently local and uncommitted at the time of this note.

## Candidate summary

### `tests/test_prompt_name_suggestion.py`
- error count: **6**
- issue shape: mechanical test-stub typing cleanup
- risk: low
- blast radius: test-only
- likely fix:
  - replace the unused import probe with an accessed availability check,
  - type local stub constructor params as `object`,
  - avoid the partially unknown `super().__init__()` path in the stub.

### Candidates checked but not selected
- `tests/test_prompt_version_history_dialog.py`
  - **6** errors, but private-only GUI internals,
  - higher seam cost than a pure test-stub file
- `tests/test_user_profile_preferences.py`
  - **8** errors, mixed,
  - still viable later, but larger and includes protected manager access

## Selected next slice

**Slice name:** `tests prompt name suggestion stub typing cleanup`

### Candidate file
- `tests/test_prompt_name_suggestion.py`

### Why this slice was chosen
- same smallest error count as the next GUI file,
- stays test-only,
- avoids introducing new public GUI seams,
- matches the repo heuristic: prefer mechanical non-GUI cleanup before deeper private-usage GUI slices.

## Intended boundaries

### In scope
- `tests/test_prompt_name_suggestion.py`,
- clear the six file-level Pyright errors,
- keep runtime behavior unchanged,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production dialog changes,
- broader Qt fallback refactors,
- unrelated test cleanup outside this file.

## Proposed implementation approach

1. replace `import PySide6` guard with `importlib.util.find_spec("PySide6")`,
2. type the local stub constructor variadics as `object`,
3. make the local dialog-button-box stub initialize without an unknown `super().__init__()` path,
4. rerun:
   - `uv run pyright tests/test_prompt_name_suggestion.py`
   - `uv run pytest -q tests/test_prompt_name_suggestion.py`
   - `uv run ruff check tests/test_prompt_name_suggestion.py`
   - `uv run ruff format --check tests/test_prompt_name_suggestion.py`

## Likely files to change
- `tests/test_prompt_name_suggestion.py`
- this brief file

## Decision

This slice is the next recommended bounded test-only Pyright target after the implemented v15 toolbar cleanup.
