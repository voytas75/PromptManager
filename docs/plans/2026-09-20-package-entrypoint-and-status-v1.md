# PromptManager — Package Entrypoint and Status Currentness v1

Status: completed
Owner: PromptManager Team

## Goal

Make the distributable wheel expose a supported `prompt-manager` command and then reconcile the active operational status with the delivered `master` state.

## Problem evidence

The wheel built from `e218e37` contained `config`, `core`, `gui`, and `models`, but omitted root `main.py`, `cli/`, and console-script metadata. The documented checkout command `python -m main` therefore was not a usable front door after standard wheel installation.

`docs/STATUS.md` also still named `d443e8b` as current even though later delivered CLI commits reached `e218e37`.

## Scope

1. Package `main.py` and `cli/` in the wheel.
2. Add the official console script `prompt-manager = main:main`.
3. Include `config/config.template.json` as package data so the installed module carries its default template asset.
4. Keep `python -m main` as a checkout/backward-compatible invocation.
5. Add a packaging metadata regression contract and execute a real wheel-install smoke outside the repository.
6. Update user/developer launch documentation, changelog, and the active `docs/STATUS.md` checkpoint.

## Non-goals

- No change to command parsing, GUI launch semantics, configuration precedence, or prompt storage.
- No dependency upgrade or CI workflow migration.
- No config-directory migration; standard installed usage can still use `PROMPT_MANAGER_CONFIG_JSON` when a custom config path is needed.

## Acceptance criteria

- A built wheel contains `main.py`, `cli/parser.py`, and `entry_points.txt` defining `prompt-manager = main:main`.
- A fresh venv outside the repository installs the built wheel and `prompt-manager --help` exits successfully.
- Existing `python -m main --help` behavior stays valid.
- Active status identifies the current delivered revision/contracts rather than `d443e8b` as current.

## Completion update

Completed 2026-09-20:

- Added the official `prompt-manager = main:main` console entrypoint.
- Included `main.py`, `cli/`, `prompt_templates.py`, and `config/config.template.json` in the wheel.
- Kept `python -m main` as the backward-compatible checkout command.
- Made missing-checkout config creation resolve the installed config template when the working directory has no `config/config.template.json`.
- Updated README and developer launch/CLI documentation, changelog, and the active status checkpoint.

Verification completed:

- focused packaging/root-help/main-entry tests: `3 passed`;
- wheel inventory: passed for `main.py`, `prompt_templates.py`, `cli/parser.py`, config template, and `prompt-manager = main:main` entry point;
- fresh external Python 3.13 venv: installed the built wheel and `prompt-manager --help` passed;
- checkout `python -m main --no-gui --help`: passed;
- `uv lock --check` and installed dependency check: passed;
- Ruff lint/format, strict Pyright for changed source/tests, and `git diff --check`: passed.

The initial fresh-venv command used `python -m pip`, but uv-created venvs do not include pip; the valid `uv pip install --python <venv>/bin/python <wheel>` path was then used successfully.
