# PromptManager — Random prompt CLI v1

**Status:** completed — 2026-09-21
**Owner:** PromptManager Team

## Goal

Add one read-only CLI command that displays a randomly selected prompt from the local repository.

## Scope

- Add `prompt-random` to the root CLI help and command dispatch.
- Select one prompt from the existing repository list without mutating storage or invoking providers.
- Reuse the existing readable `prompt-show` text renderer.
- Return a clear empty-library result when there are no prompts.

## Out of scope

- Search/ranking changes, random execution, persistence, GUI work, filters, JSON/full variants, or provider calls.

## Done criteria

- A deterministic focused test proves one selected prompt renders through the standard prompt-detail text view.
- An empty repository produces a clear non-error response.
- Focused tests, Ruff, Pyright on touched paths, and diff checks pass.

## Completion update

**Status: completed — 2026-09-21**

- Added `prompt-random`, a read-only local command that samples the existing repository list and reuses the normal `prompt-show` text formatter.
- The command makes no provider calls and changes neither prompts nor activity records.
- An empty catalog returns `No prompts available. Add or import a prompt first.` with exit code `0`.

Verified:

```bash
.venv/bin/pytest tests/test_main_entry.py::test_prompt_random_command_displays_one_repository_prompt \
  tests/test_main_entry.py::test_prompt_random_command_reports_empty_repository \
  tests/test_main_entry.py::test_prompt_show_command_outputs_prompt_details \
  tests/test_main_entry.py::test_prompt_find_command_lists_matching_prompts -q
# 4 passed

.venv/bin/ruff check cli/parser.py cli/commands.py tests/test_main_entry.py
.venv/bin/ruff format --check cli/parser.py cli/commands.py tests/test_main_entry.py
.venv/bin/pyright cli/parser.py tests/test_main_entry.py
# passed / 0 errors

.venv/bin/python -m main --help
# root help includes prompt-random

git diff --check
# passed
```
