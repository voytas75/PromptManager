# PromptManager — Next tests Pyright slice plan v15

Status: proposed
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after the implemented v14 cleanup of `tests/test_settings_dialog_live_preview.py`.

## Goal

Continue the private-usage cleanup lane using the same public-widget lookup pattern where possible.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v14 implementation commit `5d95792`:
- `tests/test_settings_dialog_live_preview.py` is closed,
- fresh `uv run pyright tests --stats` shows the tests backlog at **963 errors**,
- the next smallest private-only candidate is `tests/test_prompt_toolbar.py` with **5** errors,
- all five errors are direct `reportPrivateUsage` on toolbar internals:
  - `_recent_button`
  - `_new_button`
- `gui/widgets/prompt_toolbar.py` currently exposes no stable public widget names for those controls.

Latest shipped checkpoint before this pick:
- commit: `5d95792`
- commit message: `test: tighten settings dialog live preview typing`

## Candidate summary

### `tests/test_prompt_toolbar.py`
- error count: **5**
- issue shape: direct `reportPrivateUsage` on toolbar button access
- risk: low-to-moderate
- blast radius: test plus a tiny production public-surface seam
- likely fix: add stable `objectName` values for the Recent and Quick Capture buttons, then rewrite the test through `findChild(...)`

### Candidates checked but not selected
- `tests/test_prompt_name_suggestion.py`
  - **6** errors, mixed/non-private,
  - still a valid future candidate, but the current public-widget pattern looks reusable and lower-risk here
- `tests/test_prompt_version_history_dialog.py`
  - **6** errors, private-only,
  - larger than the toolbar file

## Selected next slice

**Slice name:** `tests prompt toolbar public-widget access cleanup`

### Candidate file
- `tests/test_prompt_toolbar.py`

### Why this slice was chosen
- smallest visible private-only candidate after v14,
- same general GUI pattern just worked in the settings-dialog slice,
- likely solvable with stable widget names and caller-side test rewiring.

## Intended boundaries

### In scope
- `tests/test_prompt_toolbar.py`,
- minimal supporting public widget naming in `gui/widgets/prompt_toolbar.py`,
- clear the five `reportPrivateUsage` errors,
- keep runtime behaviour unchanged,
- keep Ruff and formatting green,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- toolbar behavior changes,
- wider toolbar refactors,
- unrelated GUI private/protected cleanup outside this slice.

## Proposed implementation approach

1. add stable `objectName` values for the Recent and Quick Capture buttons,
2. rewrite the test to use `findChild(...)` with `QPushButton` / `QToolButton`,
3. keep the menu assertions through the public menu on the found button,
4. rerun:
   - `uv run pyright tests/test_prompt_toolbar.py`
   - `uv run pytest -q tests/test_prompt_toolbar.py`
   - `uv run ruff check tests/test_prompt_toolbar.py gui/widgets/prompt_toolbar.py`
   - `uv run ruff format --check tests/test_prompt_toolbar.py gui/widgets/prompt_toolbar.py`

## Likely files to change
- `tests/test_prompt_toolbar.py`
- `gui/widgets/prompt_toolbar.py`
- this brief file

## Decision

This slice is the next recommended bounded test-only Pyright target after the implemented v14 settings-dialog live-preview cleanup.
