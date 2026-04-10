# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Pyproject-only Install v1
Primary sources:
- `pyproject.toml`
- `README.md`
- `docs/README-DEV.md`

## Goal

Make `pyproject.toml` the single declared dependency source for local installation guidance and remove legacy `requirements*.txt` files.

## Scope

### In scope
- remove `requirements.txt`
- remove `requirements-dev.txt`
- update user-facing install instructions to use `pip install -e .` or `pip install -e .[dev]`
- update dependency-related error messages that still mention `requirements.txt`

### Out of scope
- dependency version changes
- lockfile introduction
- CI/workflow redesign
- packaging/backend changes

## Acceptance checks

1. No tracked `requirements*.txt` files remain.
2. README install instructions point to `pyproject.toml`-driven installs.
3. Developer docs point to `pip install -e .[dev]`.
4. Runtime GUI dependency guidance no longer mentions `requirements.txt`.

## Notes for implementation

- Keep the change mechanical and boring.
- Do not change actual dependency versions in this slice.
