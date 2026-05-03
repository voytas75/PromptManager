# PromptManager — Next tests Pyright slice plan

Status: proposed
Date: 2026-05-03
Source of truth: this note records the next selected bounded slice only; no implementation is included.

## Goal

Pick the next smallest credible `tests` Pyright reduction slice after the recently completed:
- GUI Pyright slice
- fixture-heavy strict typing slice

## Current verified context

Confirmed by `uv run pyright tests --stats` on 2026-05-03:
- remaining tests backlog: **1053 errors**
- backlog is still mixed across several categories:
  - fixture typing / unknown parameters
  - dynamic stub typing
  - `reportPrivateUsage` in GUI-facing tests
  - a few operator / optional access issues

Confirmed smallest clear next seam from the fresh stats review:
- `tests/test_analytics_panel_gui.py`
- roughly **29 errors** concentrated in one file
- dominant pattern: unknown `_args` / `_kwargs` in stubs and partially unknown list/dict return types

This is a better next slice than starting with the large `reportPrivateUsage` clusters because:
- it is single-file
- it is mechanical
- it stays inside test-only code
- it should not require product or runtime redesign

## Selected next slice

**Slice name:** `tests analytics panel gui strict typing`

### Candidate file
- `tests/test_analytics_panel_gui.py`

### Why this slice was chosen
- smallest obvious clustered file from the fresh `tests --stats` output
- bounded mechanical patterns already visible in the file
- low expected blast radius
- likely verification can stay local

## Expected error patterns in this slice

From the current file inspection and Pyright output:

1. stub methods return untyped containers
   - `list(self)` returns `[]`
   - `get_prompt_execution_statistics(...)` returns `{}`
   - `get_model_usage_breakdown(...)` returns `[]`
   - `get_benchmark_execution_stats(...)` returns `[]`

2. stub methods accept untyped variadics
   - `*_args`
   - `**_kwargs`

3. manager helpers use untyped signatures
   - `get_execution_analytics(...)`
   - `diagnose_embeddings(...)`

4. likely follow-up cleanup
   - make return types explicit enough for `AnalyticsDashboardPanel`
   - keep the existing test behavior unchanged

## Intended boundaries

Keep this slice narrow.

### In scope
- typing only in `tests/test_analytics_panel_gui.py`
- explicit parameter annotations for stub methods
- explicit return annotations for stub methods
- local casts/protocol-style helpers only if strictly needed inside this file

### Out of scope
- changing production GUI code
- changing analytics dashboard behavior
- broad shared fixture redesign
- tackling `reportPrivateUsage` clusters
- touching other analytics tests unless an unavoidable dependency is discovered

## Proposed implementation approach

1. annotate the stub repository methods with concrete return types
   - empty lists should become typed lists
   - empty dicts should become typed dicts

2. annotate variadic stub parameters explicitly
   - use narrow `object`/`Any` only where needed
   - prefer signatures that match actual call shape when obvious

3. annotate manager stub methods and keep `TokenUsageTotals` path explicit

4. rerun file-level Pyright immediately

5. if green, verify only the matching local test target first

## Likely files to change
- `tests/test_analytics_panel_gui.py`

## Verification plan for the future implementation step

Minimum intended verification after implementation:
- `uv run pyright tests/test_analytics_panel_gui.py`
- `uv run pytest -q tests/test_analytics_panel_gui.py`
- `uv run ruff check tests/test_analytics_panel_gui.py`
- `uv run ruff format --check tests/test_analytics_panel_gui.py`

Optional follow-up metric after the slice lands:
- `uv run pyright tests --stats`

## Risks / do weryfikacji

### Confirmed
- the file is a clean single-file cluster
- the dominant issue type is strict typing, not product logic

### Do weryfikacji
- exact concrete return types expected by `AnalyticsDashboardPanel` methods
- whether a local protocol/helper alias is needed to avoid partial unknown cascades
- whether file-level green leaves any hidden import-time dependency edge into another test helper

## Decision

The next slice is intentionally **not** a broad `private-usage` cleanup.

The next slice to implement later should be:

> `tests/test_analytics_panel_gui.py` strict typing cleanup

This note is SSOT for the selected next bounded tests slice until superseded by a newer plan.