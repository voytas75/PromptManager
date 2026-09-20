# PromptManager — Root CLI Help Card v1

Status: completed
Owner: PromptManager Team

## Goal

Replace the hard-to-scan stock `argparse` root help with a compact operator card.

## Problem evidence

The existing root command used the entire subcommand list in both `usage:` and the
flat positional-arguments section. On 2026-09-20 its rendered output had 64 lines,
a 426-character usage line, and 23 wrapped subcommand descriptions.

## Scope

1. Keep command parsing, names, handlers, and individual `<command> --help` output unchanged.
2. Render root usage as `main.py [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]`.
3. Use a root-only formatter at width 110 with a 34-character command-name column.
4. Group every public command once under functional headings.
5. Present root options as Global options; retain `--gui` compatibility but omit it from the card because GUI is already the default.
6. Add tests for the root card contract, update the CLI developer guide and changelog.

## Non-goals

- No command rename, alias, or option semantics change.
- No parser/dispatch refactor.
- No change to individual command help output.
- No colour, paging, or interactive command picker.

## Acceptance criteria

- Root help has no expanded all-command `usage:` list.
- Each public command occurs once in the grouped command card.
- Long command names, including `prompt-restore-version` and `prompt-chain-validate`, share their description line.
- Root help exposes only useful global controls and directs users to `COMMAND --help`.
- Existing CLI/help contract tests and relevant quality gates pass.

## Completion update

Completed 2026-09-20:

- Replaced the stock root `argparse` output with a grouped operator card.
- Collapsed usage to `main.py [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]`.
- Grouped all public commands exactly once under catalog, lifecycle, recommendations, chains, and operations headings.
- Used a 34-character command column and wrapped only descriptions, preserving long command names on their own description line.
- Retained parser names, dispatch, root `--gui` compatibility, and individual command help behavior; the root card intentionally omits `--gui` because launching the desktop app remains default.
- Updated the CLI developer guide and changelog.

Verification completed:

- root help contract suite: `3 passed`;
- root CLI rendering: `54` lines, max line width `109`, no expanded all-command usage list;
- `prompt-show --help` subcommand contract: passed;
- Ruff lint/format: passed;
- strict Pyright for parser and root-help contract tests: `0 errors`;
- `git diff --check`: pending final complete gate.
