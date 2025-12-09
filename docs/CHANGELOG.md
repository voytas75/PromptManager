# Changelog

All notable changes to **Prompt Manager** will be documented in this file.

## [0.22.15] - 2025-12-09

### Fixed

- LiteLLM offline and embedding checks now surface specific `PROMPT_MANAGER_*` environment variables in warnings, fall back to deterministic embeddings when credentials are missing, and keep the GUI/CLI in a safe offline state.
- Redis cache availability no longer raises during bootstrap; the Settings dialog and maintenance Redis tab display high-contrast banners indicating disabled or error states, with refresh controls gated when the cache is unavailable.

## [0.22.14] - 2025-12-08

### Added

- Introduced comprehensive token usage tracking: repository analytics now aggregate prompt/completion/total tokens, the `HistoryPanel` renders a dedicated Tokens column plus CSV/export fields, and the workspace result header reports per-run metrics along with rolling session totals.
- Added session + overall token summaries to the main workspace label and analytics dashboard, and exposed `PromptManager.get_token_usage_totals()` for CLI/GUI consumers so `python -m main history-analytics` can print window/overall token breakdowns alongside success rates.
- Updated README/README-DEV/CLI docs to describe the new tracking surfaces, ensuring teams know where to inspect per-run costs and how the session label reflects spend without leaving the desktop app.

## [0.22.13] - 2025-12-08

### Fixed

- Silenced LiteLLM's outstanding Pydantic ConfigDict deprecation spam (tracked in [BerriAI/litellm#9731](https://github.com/BerriAI/litellm/issues/9731)) and Python 3.13's new sqlite `ResourceWarning` for GC-closed connections during pytest + coverage runs by adding scoped `filterwarnings` entries so CI logs stay clean until upstream fixes land.

## [0.22.11] - 2025-12-07

### Added

- Added Rentry as the third share provider: `core.sharing.RentryProvider` now handles CSRF/token negotiation against `https://rentry.co/api/new` (per the [official CLI README](https://github.com/radude/rentry/blob/master/README.md), retrieved 2025-12-07), surfaces edit-code management notes via the GUI share controller, registers alongside ShareText/PrivateBin during bootstrap, and includes unit tests + documentation updates so contributors know the new workflow.

## [0.22.12] - 2025-12-07

### Added

- Added Google Programmable Search as the fifth web search provider: `PromptManagerSettings` now accepts `web_search_provider="google"` along with `PROMPT_MANAGER_GOOGLE_API_KEY` and `PROMPT_MANAGER_GOOGLE_CSE_ID`, the factory wires a `GoogleWebSearchProvider` that calls `https://www.googleapis.com/customsearch/v1` (per [developers.google.com/custom-search/v1/using_rest](https://developers.google.com/custom-search/v1/using_rest), retrieved 2025-12-07), GUI settings/tooltips and CLI summaries surface the new credentials, persistence guards keep both values out of `config.json`, and tests cover JSON/env loads plus provider success/error parsing.
- Updated README, README-DEV, and `docs/web_search_plan.md` with the new env vars, provider lists, and API documentation, ensuring contributors know how to configure Google Programmable Search alongside Exa/Tavily/Serper/SerpApi and how Random fan-out incorporates whichever providers have secrets configured.

## [0.22.10] - 2025-12-07

### Added

- Added PrivateBin as the second share provider: `core.sharing.PrivateBinProvider` now encrypts prompt/result payloads with PBKDF2 + AES-256-GCM before uploading via the PrivateBin JSON-LD API ([docs](https://github.com/PrivateBin/PrivateBin/wiki/API), retrieved 2025-12-07); the GUI registers it alongside ShareText, CLI summaries and README/README-DEV document the new `PROMPT_MANAGER_PRIVATEBIN_*` settings, and unit tests cover payload construction plus error handling.

## [0.22.9] - 2025-12-07

### Added

- Added SerpApi as the fourth web search provider: `PromptManagerSettings` now accepts `web_search_provider="serpapi"` with a `PROMPT_MANAGER_SERPAPI_API_KEY`/`SERPAPI_API_KEY` secret, the factory wires a `SerpApiWebSearchProvider`, tooltip copy and the settings dialog gained a SerpApi option, CLI/settings summaries mask the new credential, runtime persistence skips the secret, and tests cover HTTP success/error flows plus env loading and JSON persistence.
- Updated README, README-DEV, and `docs/web_search_plan.md` with the SerpApi configuration details and cited [serpapi.com/search-api](https://serpapi.com/search-api) (retrieved 2025-12-07) so contributors know how to issue `GET https://serpapi.com/search` with `q`/`num`/localization parameters and parse the resulting `organic_results`.

## [0.22.8] - 2025-12-07

### Added

- Added Serper.dev as the third web search provider: `PromptManagerSettings` now accepts `web_search_provider="serper"` along with a `PROMPT_MANAGER_SERPER_API_KEY`/`SERPER_API_KEY` secret, the factory wires a `SerperWebSearchProvider`, and random fan-out rotates between whichever of Exa/Tavily/Serper have credentials at runtime (tests cover HTTP success/error flows and configuration loading).
- Extended the Settings dialog Integrations tab, runtime settings service, GUI tooltips, persistence guards, CLI summaries, and `.env` helpers so Serper keys can be managed in-memory alongside Exa/Tavily without ever touching disk.
- Updated README, README-DEV, and `docs/web_search_plan.md` with the new provider, env var matrix, and API references (Elastic AutoGen tutorial + Rui Ramos Serper quickstart) so contributors know how to request `https://google.serper.dev/search` with `X-API-KEY` headers and parse the `organic` documents.

## [0.22.7] - 2025-12-07

### Added

- Added Tavily as the second web search provider: `PromptManagerSettings` now accepts `web_search_provider="tavily"` plus a `PROMPT_MANAGER_TAVILY_API_KEY`/`TAVILY_API_KEY` secret, the factory builds a Tavily-backed provider when selected, and `python -m main --print-settings` surfaces masked credentials for both providers.
- Extended the Settings dialog Integrations tab, runtime settings service, CLI summaries, and persistence guards so operators can enter/manage Tavily keys in-memory alongside Exa.
- Documented the new provider in README/README-DEV, refreshed the web search integration plan with Tavily API details (`docs.tavily.com/documentation/api-reference/endpoint/search`, retrieved 2025-12-07), and updated `.codex` knowledge for future contributors.
- Added Tavily-focused unit tests covering provider parsing, HTTP failure handling, and configuration loading to keep parity with the existing Exa coverage.
- Introduced a "Random" web search provider option that rotates calls between Exa and Tavily whenever both keys are configured (and falls back to the available provider), including GUI/CLI copy and tests documenting the behaviour.

## [0.22.6] - 2025-12-06

### Changed

- Simplified prompt chains to accept a single plain-text input that automatically flows through each step: the Chain tab now drops JSON variable editors and schema panels in favour of a persistent text field, the editor trims away per-step templates/conditions, and the CLI replaces `--vars-*` flags with `--input`/`--input-file`.
- Chain execution results and streaming previews now display the raw input text rather than JSON payloads, and the CLI/GUI both reflect the new failure-handling flag per step (defaulting to “stop chain”).

## [0.22.5] - 2025-12-05

### Added

- Prompt chain executions now share the default-on “Use web search” enrichment: the Chain tab exposes a persisted checkbox, the CLI gained a `--no-web-search` override, and the backend injects provider snippets ahead of every step when a web search service is configured.

## [0.22.4] - 2025-12-05

### Added

- Embedded the Prompt Chain manager directly inside the main window as a **Chain** tab (next to Template) so operators can review/import/run chains without opening a separate dialog.

### Removed

- Removed the redundant Prompt Chains toolbar button now that the Chain tab is always visible.

## [0.22.3] - 2025-12-04

### Added

- Captured the staged Exa integration plan in `docs/web_search_plan.md` and implemented a provider-agnostic web search service with an Exa-backed provider plus PromptManager wiring so future workflows can issue live searches.
- Extended the configuration layer, CLI summary, runtime settings service, and persistence guards with `web_search_provider`/`EXA_API_KEY` awareness to keep credentials in memory only while still surfacing provider health in `python -m main --print-settings`.
- Added an **Integrations** tab to the Settings dialog so operators can pick the active provider and enter their Exa API key without editing environment variables; values flow through the runtime settings service and remain transient on disk.
- Workspace toolbar now includes a default-on “Use web search” checkbox; when enabled and Exa is configured, prompt executions automatically fetch the top summaries/highlights from the web and prepend them to the request so runs include fresh context. Uncheck it to force offline-only behaviour per run.

### Changed

- Workspace web context enrichment now includes every available snippet/highlight returned by the provider and, whenever the combined snippets exceed ~5,000 words, compacts them via the configured fast LiteLLM model before prepending to prompt executions.

## [0.22.2] - 2025-12-04

### Added

- Introduced a native Prompt Chain editor dialog so operators can create, edit, and delete chains inside the GUI, complete with schema fields, ordered step management, and per-step condition controls.
- Wired the Prompt Chains manager to the new editor, added toolbar actions for New/Edit/Delete, and documented the workflow so teams no longer need to hand-edit JSON for routine changes.

## [0.22.1] - 2025-12-04

### Added

- Added a **Prompt Chains** toolbar button and management dialog so operators can review chain descriptions, steps, and variable schemas inside the GUI, import JSON definitions, and run chains with busy-indicator wrapped execution plus toast-confirmed refresh/import flows.
- Introduced a shared `chain_from_payload` helper that powers both the CLI (`prompt-chain-apply`) and the new GUI import action, ensuring JSON validation rules stay in sync across surfaces.

## [0.22.0] - 2025-12-04

### Added

- Introduced prompt chain data models, repository storage, and a PromptManager mixin so multi-step workflows can sequence prompts with JSON-schema validated variables, per-step history logging, and notification tracking.
- Added `prompt-chain-list`, `prompt-chain-show`, `prompt-chain-apply`, and `prompt-chain-run` CLI commands to manage chain definitions from JSON files, inspect step wiring, and execute chains with inline or file-provided variables.
- Updated README and developer guide with prompt chaining guidance, including JSON templates and feature overviews, ensuring teams can deploy the new automation workflow without reading code.

## [0.21.0] - 2025-12-03

### Added

- Added a LiteLLM text-to-speech model configuration (GUI + JSON/env support) so upcoming voice playback flows can target dedicated TTS endpoints without overloading the fast/inference chat models.
- Added a speaker button to the workspace result overlay that streams LiteLLM text-to-speech audio for the last run whenever `PROMPT_MANAGER_LITELLM_TTS_MODEL` and Qt Multimedia support are available.
- Added a LiteLLM TTS streaming toggle (default on) so audio playback can begin while LiteLLM streams bytes; fall back to full-file playback by setting `PROMPT_MANAGER_LITELLM_TTS_STREAM=false`.

## [0.20.0] - 2025-11-29

### Added

- Introduced the **Enhanced Prompt Workbench**: a dedicated modal workspace with a guided wizard, block palette, improved template editing, template variable dialogs, CodexExecutor Brainstorm/Peek/Run loops, and live history/feedback logging. Launch it via the new 🆕 toolbar button to scaffold prompts from scratch, remix existing templates, or iterate through runs before exporting into the catalogue.

### Changed

- Template preview widget now exposes programmatic variable population/refresh hooks so the Workbench (and future tooling) can keep placeholder forms in sync without user typing.
- Workbench runs accept empty “Test input” fields by reusing sample variable values or the prompt goal, keeping iterative flows moving without boilerplate text.

## [0.19.1] - 2025-11-29

### Added

- Added a dedicated Prompt Template Editor dialog (available from the main toolbar) with per-workflow editors, inline validation, and reset controls so operators can adjust LiteLLM system prompts without opening the full settings surface.

### Changed

- Settings application logic now supports partial updates, enabling focused dialogs (like the new template editor) to persist their data without clobbering unrelated runtime options.

## [0.19.0] - 2025-11-28

### Added

- Added an **Analytics** tab to the GUI that visualises usage frequency, LiteLLM model cost breakdowns, benchmark success, and intent success trends with switchable bar/line charts, adjustable look-back windows, and one-click CSV export; embedding health summaries now live alongside the charts.
- Extended the `diagnostics` CLI with an `analytics` target plus `--window-days`, `--prompt-limit`, `--dataset`, and `--export-csv` flags so terminal users can build the same dashboard snapshot (including embedding health) and persist datasets to CSV.
- Introduced a `core.analytics_dashboard` helper and new repository aggregate queries for model usage/benchmark stats, enabling future surfaces to reuse the same execution/benchmark/embedding metrics.

### Changed

- Diagnostics documentation and README sections now highlight both `diagnostics embeddings` and `diagnostics analytics`, clarifying how to obtain vector health checks or exportable performance snapshots.

## [0.18.0] - 2025-11-28

### Added

- Added a bordered **Share** panel beneath the prompt action buttons that lets users pick the data scope (body only, body + description, body + description + scenarios), optionally include metadata, and then choose a provider. The first provider ships via ShareText and automatically copies the resulting link to the clipboard while logging usage analytics.

### Changed

- Version bumped to `0.18.0` to reflect the new sharing workflow and UI changes.

## [0.17.9] - 2025-11-28

### Added

- Added an execution benchmark workflow powered by `PromptManager.benchmark_prompts`, a new CLI command (`python -m main benchmark …`), and summary output that surfaces duration/token usage plus existing history analytics so prompt authors can compare models without leaving the terminal.
- Added scenario refresh APIs (`PromptManager.refresh_prompt_scenarios`) with matching CLI (`python -m main refresh-scenarios <prompt_id>`) and a **Refresh Scenarios** button in the prompt detail pane so stored usage examples stay in sync with the latest LiteLLM suggestions.
- Extended the maintenance dialog with a **Category Health** table driven by new repository analytics, exposing per-category prompt counts, activation ratios, success rates, and last execution timestamps for rapid taxonomy audits; README/README-DEV now document the panel.
- Added a **Create Backup Snapshot** control to the maintenance dialog plus a `PromptManager.create_data_snapshot` helper so admins can zip the SQLite DB, Chroma directory, and a manifest before running destructive reset tasks; README/README-DEV document the workflow.

### Added

- ResponseStyle records now include a `prompt_part` classification so entries can represent response styles, system snippets, output formatters, or any other prompt component; the SQLite schema, repository helpers, and tests were updated accordingly.

### Changed

- The GUI Response Styles tab has been renamed to **Prompt Parts**, the dialog now captures the prompt part type, and detail/export views display the new metadata so teams can curate all prompt fragments from a single surface; README/README-DEV were refreshed to document the workflow.

## [0.17.8] - 2025-11-27

## [0.17.7] - 2025-11-26

### Added

- Added a pinned `pytest-xdist` dev dependency so the default `pytest -n auto` workflow can run under the project's managed environment without relying on globally installed packages.

## [0.17.6] - 2025-11-25

### Added

- Introduced a reusable `core.templating` module that wraps Jinja2 with strict undefined handling, custom `truncate`/`slugify`/`json` filters, and optional JSON Schema or Pydantic validation helpers so any surface can safely render prompt templates.
- Added a live **Template Preview** pane to the GUI workspace that renders the selected prompt with pasted JSON variables, validates payloads against optional schemas, and highlights missing/invalid fields inline before execution.
- Documented the new workflow in README/README-DEV and bundled `Jinja2` plus `jsonschema` in the dependency manifests.

## [0.17.5] - 2025-11-24

### Added

- Prompt description generation now falls back to a deterministic summary (built from name, category, tags, and a trimmed context snippet) whenever LiteLLM is unavailable or errors, keeping workflows moving even without external API access.
- History logging captures richer execution context—prompt metadata, runtime config (model, streaming, conversation length), and future response-style details—by extending `HistoryTracker.record_success` with a structured `context` payload.

## [0.17.4] - 2025-11-23

### Added

- Added a **Similar** action to the prompt list context menu that runs an embedding-based lookup and populates the prompt catalogue view with recommendations, helping users discover existing solutions before adding duplicates.
- Introduced a **Body size (long-short)** sort option so catalogues can be ordered by prompt body length when hunting for comprehensive templates or trimming oversized entries.
- Added a **Rating (high-low)** sort so teams can bubble the highest-rated prompts to the top when sharing favourites during reviews or audits.

## [0.17.3] - 2025-11-23

### Added

- Surfaced a **Prompt Templates** tab in the settings dialog that lists every LiteLLM system prompt (name, description, scenario, refinement) with inline editors, so teams can tailor the guidance without editing code.
- Added Reset and Reset-all controls that restore the baked-in defaults with a single click, making it safe to experiment with custom templates.
- Persist prompt template overrides via settings/configuration, propagate them through the factory, and ensure LiteLLM helpers respect the overrides immediately without restarting the app.

## [0.17.2] - 2025-11-22

### Changed

- Execute-as-context now remembers the most recent task description across sessions, pre-filling the dialog with whatever the user last entered so repetitive workflows stay fast even after restarting the app.

## [0.17.1] - 2025-11-22

### Added

- Added a **Clear** control to the workspace toolbar that immediately resets the pasted code form, Output tab, and chat transcript, ensuring users can begin a new workflow without manually clearing multiple panes.
- Updated README guidance so the new action is documented alongside Detect/Suggest/Run workflows.

## [0.17.0] - 2025-11-22

### Added

- Introduced a first-class **PromptCategory** data model with SQLite persistence, a `prompt_categories` table, and repository CRUD helpers, enabling teams to manage taxonomies without editing code.
- Added a `CategoryRegistry` that seeds defaults, loads optional JSON overrides (`PROMPT_MANAGER_CATEGORIES_PATH`/`PROMPT_MANAGER_CATEGORIES`), and exposes manager APIs for listing/creating/updating/toggling categories.
- Extended the GUI with a **Manage** button beside the category filter that opens the new Category Manager dialog (create/edit/archive flows plus inline validation).

### Changed

- Prompts now store a `category_slug` alongside the human-readable label; legacy records are backfilled automatically and prompt filters use the slug for stability.
- Prompt creation/update flows normalise categories through the registry so renames/archives propagate instantly across the catalogue.
- README and settings documentation now cover the taxonomy workflow and configuration hooks.

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
