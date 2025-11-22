# Changelog

All notable changes to **Prompt Manager** will be documented in this file.

## [0.16.0] - 2025-11-22

### Removed

- Retired the Task Template feature entirely: removed the dataclass, SQLite schema, repository/manager APIs, GUI controls, dialogs, telemetry events, and tests. Quick actions now serve as the single way to seed the workspace with starter text, and documentation/settings have been updated accordingly.

## [0.15.2] - 2025-11-22

### Added

- Surfaced a dedicated LiteLLM embedding model field in the settings dialog and configuration persistence so semantic search can target any provider-supported embedding endpoint without editing JSON manually.

### Changed

- Switched the default embedding backend to LiteLLM, wiring the Prompt Manager to generate real embeddings (via `text-embedding-3-large` by default) instead of the deterministic hash stub, and updated documentation/templates to reflect the new requirement.
- LiteLLM embeddings no longer fall back to the conversational fast model when no embedding model is configured, preventing accidental 400s from completion-only deployments and keeping semantic search aligned with the configured embedding endpoint.

## [0.15.1] - 2025-11-19

### Changed

- Removed the Diff tab from the GUI result pane, dropped the supporting diff preview utility/tests, and refreshed documentation so executions now focus on the Output and Chat views.

## [0.15.0] - 2025-11-22

### Added

- Introduced **Response Style** data model with SQLite persistence, CRUD APIs, and repository/manger tests covering happy-path and failure handling.
- Added a dedicated Response Style tab (adjacent to Prompts/History/Notes) with New/Edit/Delete workflows, clipboard/markdown/export helpers, and a detail pane highlighting tone, voice, and formatting guidelines for the selected preset.
- Response Style dialog now accepts a single pasted phrase and auto-generates required metadata (name, description, format instructions, initial example) on save for rapid capture.
- Added a **Notes** tab beside Prompts/History that stores one-field prompt notes with create/edit/delete actions, clipboard copy, Markdown preview, and export-to-file support powered by the new PromptNote model.

### Changed

- Updated dialogs module to include a dedicated Response Style editor that mirrors the task template UX, ensuring reusable formatting guidance can be authored without touching the database.
- Refreshed README to document the Response Style registry feature.

## [0.14.0] - 2025-11-18

### Added

- Introduced *internal modularisation* of the **core.prompt_manager** component.
  - Converted former monolithic `core/prompt_manager.py` into a package.
  - Added `core/exceptions.py` centralising shared exception classes.
  - Added facades:
    - `core/prompt_manager/storage.PromptStorage` (persistence wrapper).
    - `core/prompt_manager/execution.PromptExecutor` (LLM execution wrapper).
    - `core/prompt_manager/engineering.PromptEngineerFacade` (prompt refinement wrapper).

### Changed

- Updated `core/prompt_manager/__init__.py` to re‑export new facades and
  maintain backward compatibility for existing imports.
- Bumped project version to **0.14.0** in `pyproject.toml`.

### Fixed

- Resolved test collection error in `gui/dialogs.py` when imported outside the
  `gui` package by adding a fallback import path.

## [0.13.14] - 2025-11-05

- Packaged a default Prompt Manager icon and set it as the Qt application/window icon so Windows builds display branded taskbar and shell visuals out of the box.
- Styled chat transcripts with a tinted background on user turns to improve visual separation during ongoing conversations.
- Added a settings control for choosing the user chat bubble colour, allowing teams to align the transcript with their theme.
- Introduced a light/dark theme toggle in the settings dialog and apply palette updates across the application.
- Updated the Info dialog to show the application icon and credit Icons8 as the icon source.

## [0.13.13] - 2025-11-05

- Added a dedicated LiteLLM inference model configuration alongside the existing fast model so future workflows can route tasks to latency-appropriate endpoints.
- Updated the settings dialog with tabbed navigation and dual model inputs while keeping secret handling unchanged.
- Introduced a LiteLLM routing matrix so each workflow can target the fast or inference tier independently from the settings dialog or JSON configuration.

## [0.13.12] - 2025-11-22

- Removed the `catalog-import` CLI command while keeping the GUI import workflow; catalogue entries can still be applied from JSON via the Import button, with export remaining available via CLI and GUI.
- Simplified documentation to reflect the GUI-focused import flow and removed references to automatic catalogue seeding.
- Added a Data Reset tab to the maintenance dialog with guarded actions to clear the SQLite prompt database, wipe Chroma embeddings, or reset all application data without touching settings.

## [0.13.11] - 2025-11-22

- Added an exit toolbar icon (and `Ctrl+Q` shortcut) that closes the GUI gracefully by running shutdown hooks before quitting.

## [0.13.10] - 2025-11-22

- Added a LiteLLM streaming execution mode with optional token callbacks so prompt runs can surface incremental output while histories persist aggregated responses.
- Introduced a `litellm_stream` configuration flag, surfaced it via the CLI settings summary, and extended documentation with environment variable guidance.
- Added a GUI toggle for LiteLLM streaming and live output updates in the workspace result pane.

## [0.13.9] - 2025-11-05

- Removed explicit LiteLLM timeouts across prompt execution, engineering, metadata generation, and embedding calls so long-running requests rely on provider defaults instead of failing after fixed client deadlines.

## [0.13.8] - 2025-11-22

- Added a Duplicate Prompt action to the prompt list context menu that opens a pre-filled editor for cloning existing entries before saving.

## [0.13.7] - 2025-11-22

- Added an Info dialog to the GUI toolbar that links to the author's GitHub profile and surfaces system CPU/architecture details alongside the project's open source license summary.
- Added an Execute Prompt action to the prompt list context menu that populates the workspace text window and runs the selection immediately via the configured AI backend.

## [0.13.6] - 2025-11-22

- Added a prompt catalogue overview on the Maintenance tab with live counts for prompts, categories, tags, stale entries, and recent updates, plus a manual refresh control.
- Exposed prompt catalogue statistics via `PromptManager.get_prompt_catalogue_stats` so other tooling can reuse the aggregated metrics.

## [0.13.5] - 2025-11-22

- Added a **Copy to Text Window** control in the GUI result pane that rehydrates the latest output into the workspace editor for rapid follow-up prompts without manual copying.
- Updated documentation to reference the new control and clarify the result workflow sequence.

## [0.13.4] - 2025-11-19

- Added scenario generation to the prompt editor, including a LiteLLM-backed generator with heuristic fallback when AI assistance is disabled.
- Persist usage scenarios in SQLite and surface them alongside the prompt body in the detail pane.
- Updated the GUI to display saved scenarios beneath each prompt body and refreshed documentation to describe the workflow.

## [0.13.3] - 2025-11-02

- Stop forwarding LiteLLM `drop_params` directives to Azure/OpenAI providers; configured keys are now stripped locally before requests, preventing `Unknown parameter: 'drop_params'` failures during prompt execution.
- Adjust LiteLLM fallback retries to reuse the stripped parameter set without injecting unsupported fields back into follow-up requests.
- Added optional `litellm_reasoning_effort` configuration so reasoning-capable models (gpt-4.1, o-series, gpt-5) can request `minimal`, `medium`, or `high` effort without code changes.
- Automatically switch embeddings to LiteLLM when an `embedding_model` is configured so semantic sync uses your OpenAI embedding choice out of the box.

## [0.13.2] - 2025-11-17

- Removed the bundled default prompt catalogue and associated packaging assets; catalogue imports now require an explicit JSON file or directory.
- Removed the `catalog_path` configuration knob; startup no longer consults settings for catalogue imports, and the GUI/CLI now require explicit paths each time.
- Added an Apply button to the prompt editor so changes can be saved without closing the dialog.
- LiteLLM name/description generators now retry without unsupported parameters (e.g., `max_tokens`) when models reject them.
- Added `litellm_drop_params` configuration support and propagate it through prompt execution, metadata generation, and prompt engineering so LiteLLM can pre-drop unsupported parameters per official guidance.

## [0.13.1] - 2025-11-16

- Applied palette-aware borders to the main Prompt Manager window and settings dialog so both stay visually distinct across light and dark themes.

## [0.13.0] - 2025-11-16

- Added a Render Output action in the GUI result pane that opens LLM responses as rendered Markdown within a dedicated preview window.
- Introduced a reusable markdown preview dialog so other workflows can surface formatted content without leaving the application.
- Updated project documentation to cover the new viewing option and guide users towards the Markdown preview button.

## [0.12.0] - 2025-11-15

- Added a LiteLLM-backed prompt engineering workflow that analyses and refines prompt bodies using the new meta-prompt ruleset.
- Exposed a **Refine** button in the prompt editor dialog that applies the improved prompt and surfaces analysis, checklist items, and warnings to the user.
- Introduced unit tests covering the prompt engineering helper and manager integration, plus updated documentation to describe the new capability.

## [0.11.1] - 2025-11-15

- Removed LiteLLM API key loading from JSON configuration files to prevent accidental secret leaks; credentials must now be supplied via environment variables or a secret manager.
- Added a checked-in `config/config.template.json` with non-secret defaults and scrubbed the local example config.
- Updated the settings dialog and documentation so LiteLLM API keys are never persisted to disk and existing configs remain valid.
- Enhanced `python -m main --print-settings` to display filesystem health checks and mask LiteLLM API keys while confirming model configuration.

## [0.11.0] - 2025-11-12

- Enabled multi-turn chat continuation in the GUI with **Continue Chat** / **End Chat** controls and a dedicated chat transcript tab.
- Persisted full conversations alongside each execution record and surfaced them in the history panel for review and export.
- Extended the LiteLLM execution pipeline to accept conversation context so follow-up turns reuse prior exchanges reliably.

## [0.10.0] - 2025-11-11

- Added a reusable notification centre with task tracking helpers and GUI integration (status indicator, history dialog).
- Surfaced detailed feedback for prompt execution, name/description generation, and embedding sync operations through notifications.

## [0.9.0] - 2025-11-11

- Tracked prompt usage in a single-user profile persisted to SQLite, powering personalised categories, tags, and recent history.
- Biased intent suggestions with stored preferences so frequently used categories surface first in GUI and CLI queries.
- Added regression tests for profile persistence along with lightweight collection stubs to keep factories decoupled from Chroma.
- Softened LiteLLM/Azure content filter failures with concise messaging and fallback-friendly handling in the GUI.

## [0.8.5] - 2025-11-10

- Added a keyboard-driven command palette (`Ctrl+K` / `Ctrl+Shift+P`) with quick actions mapped to common prompt workflows.
- Introduced configurable quick action shortcuts (`Ctrl+1`–`Ctrl+4`) and prompt matching logic for explain, debug, document, and enhance flows.
- Allowed users to define custom quick actions via settings/config JSON (including hints, templates, shortcuts, and prompt IDs) and added unit tests for the palette utilities.

## [0.8.4] - 2025-11-10

- Added automatic language detection and syntax highlighting to the GUI query workspace.
- Introduced reusable detection utilities and covered them with unit tests.
- Updated documentation to describe the highlighted workspace behaviour.

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
