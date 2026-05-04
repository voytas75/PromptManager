# PromptManager — Next tests Pyright slice plan v10

Status: implemented
Date: 2026-05-04
Updated: 2026-05-04
Source of truth: this note records the next bounded `tests` Pyright slice selected after the implemented v9 cleanup of `tests/test_gui_diagnostics_status.py` and now closed by the shipped `tests/test_prompt_chain_cli.py` cleanup.

## Goal

Keep the technical quality lane moving with another ultra-small test-only Pyright slice while still avoiding the private-usage-heavy clusters.

## Current verified context

Confirmed by fresh live checks on 2026-05-04 after the v9 implementation commit `d03b8ed`:
- `tests/test_gui_diagnostics_status.py` is closed,
- `tests/test_prompt_chain_cli.py` is now also closed,
- file-level verification for the prompt-chain CLI slice is green:
  - `uv run pyright tests/test_prompt_chain_cli.py` → `0 errors`
  - `uv run pytest -q tests/test_prompt_chain_cli.py` → `2 passed`
  - `uv run ruff check tests/test_prompt_chain_cli.py` → `OK`
  - `uv run ruff format --check tests/test_prompt_chain_cli.py` → `OK`
- fresh `uv run pyright tests --stats` now shows the tests backlog reduced further from **977 errors** to **975 errors**.

Latest shipped checkpoint:
- implementation commit: do zapisania po commicie bieżącej zmiany
- prior checkpoint before this pick: `d03b8ed` (`test: tighten gui diagnostics status typing`)

## Candidate summary at selection time

### `tests/test_prompt_chain_cli.py`
- error count at pick time: **2**
- issue shape: stub-vs-`PromptManager` parameter compatibility on direct CLI handler calls
- risk: low
- blast radius: test-only
- likely fix: keep the stub local and narrow the callsite typing with an explicit cast rather than widening production CLI signatures

### Candidates checked but not selected
- `tests/test_prompt_manager_reset.py`
  - also **2** errors,
  - but both are `reportPrivateUsage` on `_create_calls`,
  - still intentionally deferred with the other private-usage-heavy files
- `tests/test_settings_dialog_live_preview.py`
  - **3** errors,
  - bounded but GUI-facing and less minimal than the CLI file
- `tests/test_template_preview_widget.py`
  - **3** errors,
  - also GUI-facing, lower priority than the test-only CLI seam

## Implemented slice

**Slice name:** `tests prompt chain cli strict-typing cleanup`

### Candidate file
- `tests/test_prompt_chain_cli.py`

### Why this slice was chosen
- smallest currently visible non-private candidate after v9,
- fully test-local fix path,
- lower-risk than moving into GUI files or private-usage policy decisions,
- production CLI signatures could stay unchanged.

## Implemented approach

1. confirmed the two live errors were only on the direct handler callsites,
2. added `cast` import,
3. wrapped the local test stub at the two handler invocations with `cast("Any", manager)`,
4. kept the change test-only and runtime-neutral,
5. reran the full file-level verification loop and confirmed the slice was green.

## Intended boundaries

### In scope
- `tests/test_prompt_chain_cli.py` only,
- clear the two live `reportArgumentType` errors,
- keep runtime behaviour unchanged,
- keep Ruff and formatting green,
- verify with file-level pyright/pytest/ruff only.

### Out of scope
- production CLI type-signature redesign,
- prompt-chain execution refactors,
- private-usage clusters,
- broader GUI test typing cleanup.

## Files changed for the slice
- `tests/test_prompt_chain_cli.py`
- this brief file

## Result

This slice is closed and shipped locally. The next work should be selected from a fresh post-v10 repick before continuing the tests Pyright lane.
