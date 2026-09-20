# PromptManager — CLI Help Contract Repair v1

Status: completed
Owner: PromptManager Team
Scope: documentation/help contract repair only; no CLI namespace redesign and no error-channel normalization.

## Goal

Align the documented and runtime-discovered CLI contract with the verified implementation for prompt lookup and `prompt-add` payloads, then make the developer CLI index complete enough to navigate all public commands.

## Confirmed baseline

- `cli/parser.py` exposes 26 commands and `cli/commands.py::COMMAND_SPECS` dispatches the same 26 commands.
- Root help and all leaf `--help` paths return exit `0` with stdout-only help.
- `prompt-show` resolves a UUID or one exact prompt name, but its help says UUID only.
- `README.md` and `examples/prompt-import-example.json` show `prompt_text`, while the catalog importer stores the prompt body from `context`.
- `docs/README-DEV.md` lists 17 of 26 CLI commands; it omits the current prompt read/import and several prompt-chain inspection/export commands.

## Execution order

1. Add focused regressions that lock:
   - `prompt-show --help` wording for UUID-or-exact-name resolution;
   - the checked-in `prompt-add` sample against the importer-visible `context` body contract;
   - developer CLI index coverage for all 26 parser commands.
2. Repair the smallest contract surfaces:
   - `cli/parser.py` help wording for `prompt-show`;
   - `README.md` and `examples/prompt-import-example.json` payload field name;
   - `docs/README-DEV.md` CLI table coverage and a clear runtime-help discovery pointer.
3. Validate:
   - focused regression tests;
   - real `python -m main --help` and `prompt-show --help` subprocess checks;
   - JSON parsing of the checked-in sample;
   - `ruff`, CI-scope Pyright, `git diff --check`.

## Out of scope

- Renaming `prompt-add --json` despite its input-oriented meaning.
- Uniform JSON/error-channel behavior across all CLI handlers.
- New command groups, aliases, API endpoints, provider calls, configuration changes, commits, or pushes.

## Risks and stop condition

- The examples directory is generally ignored, but the existing sample is tracked; modify only this tracked file.
- Stop and reassess if a sample payload must preserve `prompt_text` for an external compatibility contract not implemented by the importer.

## Completion update

Completed 2026-09-20:

- Added regressions for `prompt-show` help, the checked-in `prompt-add` sample's importer-visible body, and full public-command coverage in the developer CLI index.
- Corrected `prompt-show` help to advertise UUID-or-exact-name resolution.
- Corrected README and checked-in sample payloads from `prompt_text` to `context`.
- Added the nine missing public commands to `docs/README-DEV.md` and a runtime-help discovery pointer.
- Added the bounded change note to `docs/CHANGELOG.md`.

Verification completed:

- focused help/name tests: 2 passed;
- CLI help/documentation contract tests: 2 passed;
- real root and `prompt-show` help smoke plus JSON sample parse: passed;
- Ruff lint/format checks: passed;
- CI-scope Pyright (`main.py config models`): 0 errors;
- `git diff --check`: passed.

Deferred unchanged: uniform runtime error channels and an input/output rename for `prompt-add --json`.
