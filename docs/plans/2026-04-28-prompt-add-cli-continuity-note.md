# Prompt-add CLI continuity note

**Status:** active reference
**Updated:** 2026-04-28
**Scope:** capture/ingest ergonomics for CLI-first prompt creation

## Why this note exists

This note records the current shipped state of the PromptManager CLI prompt-ingest path so future work does not rely on chat memory or re-open already delivered slices.

## Confirmed shipped state

The operator-facing CLI path for adding normal prompts is now centered on `prompt-add`, which stays a thin wrapper over the existing catalog importer backend.

Delivered input modes:
- positional `PATH` for a JSON file or directory import
- inline one-prompt creation via `--name`, `--description`, `--prompt-text`
- single-payload JSON string via `--json`
- stdin payload via `--from-stdin`
- explicit single-file alias via `--input-file`

Delivered behavior and guardrails:
- `--dry-run` remains the preview/diff path
- `--no-overwrite` keeps existing same-name prompts unchanged
- exactly one input source is accepted at a time
- malformed payloads now fail fast before importer execution with explicit validation errors
- runtime behavior still reuses the existing `core/catalog_importer.py` path instead of introducing parallel CRUD logic

## Verification snapshot

Latest confirmed verification in this cycle:
- `pytest -q` → `680 passed, 1 skipped`
- `ruff check .` → passed
- `pyright main.py config models` → `0 errors`
- runtime check confirmed invalid `prompt-add --json` payloads exit with code `2` and a clear validation message

Latest shipped commit for this CLI layer:
- `88279e5` — `feat: validate prompt-add payloads early`

## What is not open anymore

The following slices are considered delivered for the current CLI ingest layer:
- public `catalog-import` return path for normal prompt JSON import
- operator-facing `prompt-add` alias
- inline prompt creation
- JSON-string and stdin input modes
- explicit `--input-file` alias
- pre-import payload validation with readable CLI errors

Do not reopen these as pending unless runtime or tests contradict this note.

## Likely next follow-ups

These are candidates, not committed scope:
1. support multi-entry list payloads more explicitly in `prompt-add --json`
2. extend validation for optional structured fields if operator errors show up in practice
3. add further CLI ergonomics only if they reduce ingest friction without duplicating importer logic

## Decision rule for continuation

For the next slice, prefer work that improves prompt-asset ingest clarity or lowers friction without creating a second import path.

If a proposed CLI change duplicates importer behavior instead of wrapping it, reject it.
