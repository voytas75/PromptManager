# Changelog

All notable changes to **Prompt Manager** will be documented in this file.

## [0.8.3] - 2025-11-10

- Added an Output/Diff view in the GUI result panel to compare generated text with the original request.
- Introduced a shared diff preview helper so other surfaces can reuse the same comparison logic.
- Covered the diff helper with unit tests to guard against regressions.
- Updated documentation to reflect the new diff viewer workflow.

## [0.8.2] - 2025-11-09

- Added prompt execution rating workflow in the GUI, storing per-run ratings and surfacing averaged quality scores in prompt details and filters.
- Persisted rating aggregates in SQLite with automatic migrations and updated CLI/manual saves to refresh prompt quality metrics.
- Extended history views, exports, and analytics logging to include captured ratings.

## [0.8.1] - 2025-11-09

- Wrapped the GUI prompt detail pane in a scrollable container to respect compositor size constraints on Wayland and avoid crashes when maximising the window with long prompt bodies selected.

## [0.7.0] - 2025-11-07

- Added configurable embedding backends (LiteLLM and sentence-transformers) with deterministic fallback and new settings/env variables.
- Introduced GUI workspace actions for intent detection, prompt suggestion, and clipboard copying.
- Split optional dependencies into extras (`cache`, `llm`, `embeddings`, `dev`) and refreshed installation docs/requirements.
- Added `python -m main suggest` CLI command to verify semantic retrieval results from the configured embedding backend.
- Logged intent workspace interactions to `data/logs/intent_usage.jsonl` for lightweight UX analytics.
- Added `python -m main usage-report` command to summarise logged workspace activity for feedback review.
- Auto-generate prompt names and descriptions via LiteLLM when the creation dialog fields are left blank.
- Added LiteLLM API version wiring (including Azure `AZURE_OPENAI_API_VERSION` alias) across settings, manager wiring, and generators.

## [0.8.0] - 2025-11-08

- Added LiteLLM-backed prompt execution with automatic history logging to the new `prompt_executions` table.
- Introduced `HistoryTracker` utilities and repository APIs for retrieving recent execution history per prompt or globally.
- Extended `PromptManager` with an `execute_prompt` workflow, public history accessors, and error handling for unavailable LiteLLM credentials.
- Updated the PySide6 GUI with **Run Prompt** / **Copy Result** actions and a result pane that surfaces execution metadata.
- Added a GUI execution history dialog showing recent runs, durations, and failure details with copyable request/response payloads.
- Added **Save Result** controls that let users annotate and persist the latest response manually, plus a filterable history tab with inline note editing and CSV export.
- Recorded prompt executions in analytics via the GUI usage logger and expanded unit tests to cover repository, history, and manager execution flows.

## [0.6.0] - 2025-11-06

- Added a lightweight intent classifier that biases semantic search results and surfaces top recommendations directly in the GUI.
- Introduced `catalog-import` and `catalog-export` CLI commands with diff previews, optional dry-run mode, and JSON/YAML export support.
- Extended the GUI toolbar with Import/Export actions and a catalogue diff dialog that mirrors the CLI workflow.
- Implemented structured catalogue diff planning to guard against accidental overwrites and provide richer logging.
- Added regression tests for catalogue diff/export, the new CLI commands, and intent-aware prompt suggestions.

## [0.5.0] - 2025-11-05

- Import the packaged prompt catalogue (or a user-supplied JSON directory) into SQLite and ChromaDB on startup.
- Add category, tag, and minimum-quality filters to the GUI alongside seeded prompt metadata.
- Provide LiteLLM-backed prompt-name suggestions derived from context when creating entries.
- Added in-app Settings dialog to configure catalogue path and LiteLLM credentials with persistence to `config/config.json`.
- Clarify the prompt editor by renaming the context field to “Prompt Body” in dialogs and detail views.
- Expose configurable `catalog_path` setting and environment variable override.
- Package the built-in catalogue for distribution and provide catalogue import helpers with unit tests.

## [0.4.1] - 2025-11-05

- Promote GUI launch to the default `python -m main` behaviour with `--no-gui` opt-out.
- Add graceful shutdown hooks for the `PromptManager` background worker and backends.
- Split GUI requirements into an optional extra (`pip install .[gui]`) and dedicated requirements file.
- Suppress noisy Chroma PostHog telemetry while keeping collection initialisation intact.
- Improve packaging metadata and configure setuptools package discovery for `pip install .`.
- Extend tests to cover GUI dependency fallbacks, offscreen detection, and manager shutdown.
