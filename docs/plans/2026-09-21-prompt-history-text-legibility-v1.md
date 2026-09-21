# PromptManager — Prompt history text legibility v1

**Status:** completed — 2026-09-21
**Owner:** PromptManager Team

## Goal

Make text-mode `prompt-history` execution records readable when request, response, or error text spans multiple lines.

## Confirmed baseline

- `cli/commands.py::run_prompt_history` currently renders execution text inline as `request: ...` and `response: ...`.
- Long or multiline request/response bodies visually merge in terminal output.
- `prompt-history --json` already provides structured output and must remain unchanged.
- GUI `HistoryPanel` already renders Request and Response as separate detail sections; this slice targets CLI text output only.

## Scope

- Add stable ASCII record separators (`====`) and named content blocks (`--- Request ---`, `--- Response ---`, `--- Error ---`) in CLI text output.
- Preserve the existing execution metadata and all JSON semantics.
- Add one focused text-output regression.

## Out of scope

- Truncation, `--full`, new flags, persistence changes, GUI layout changes, or JSON changes.

## Done criteria

- Consecutive execution records and multiline request/response/error bodies have visible boundaries in text mode.
- Existing `prompt-history --json` contract remains unchanged.
- Focused CLI tests, Ruff, Pyright on touched paths, and diff checks pass.

## Completion update

**Status: completed — 2026-09-21**

- Text-mode `prompt-history` now renders every result in a distinct `==== Execution N ====` block.
- Metadata stays on its own line; request, response, and error values each receive a separate named ASCII block.
- Multiline content is preserved verbatim inside its section rather than being collapsed or truncated.
- The JSON path was not modified.

Verified:

```bash
.venv/bin/pytest tests/test_main_entry.py::test_prompt_history_command_outputs_recent_execution_summary \
  tests/test_main_entry.py::test_prompt_history_command_outputs_json_payload \
  tests/test_main_entry.py::test_prompt_history_command_filters_by_status_and_window_days -q
# 3 passed

.venv/bin/ruff check cli/commands.py tests/test_main_entry.py
.venv/bin/ruff format --check cli/commands.py tests/test_main_entry.py
.venv/bin/pyright tests/test_main_entry.py
# passed / 0 errors

git diff --check
# passed
```
