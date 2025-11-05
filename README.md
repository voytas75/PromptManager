# Prompt Manager

Prompt Manager is a desktop-focused application for cataloguing, searching, and executing AI prompts. The project follows the architecture captured in `PromptManagerBlueprint.txt` and targets PySide6 for the GUI while relying on SQLite for canonical storage, ChromaDB for semantic retrieval, and Redis for responsive caching.

## Current Status

- Core data model defined in `models/prompt_model.py` with an extensible schema (`ext1`–`ext5`) and serialization helpers.
- `PromptRepository` (`core/repository.py`) persists prompts to SQLite and exposes CRUD operations.
- `PromptManager` service (`core/prompt_manager.py`) coordinates SQLite persistence, Redis-backed caching, and ChromaDB semantic search.
- Intent-aware search now classifies queries (debug/refactor/enhance/etc.) to bias retrieval and surface top recommendations inline in the GUI.
- Single-user preference profile captures prompt usage to personalise ranking and quick suggestions without multi-account overhead.
- Notification center tracks long-running LLM and embedding tasks and surfaces progress via the GUI status bar with a history dialog.
- Typed configuration loader in `config/settings.py` validates paths/TTL values and can hydrate from environment variables or a JSON file.
- Initial PySide6 GUI accessible via `--gui`, offering list/search/detail panes with create/edit/delete dialogs.
- Prompt browser includes sorting controls (name, quality, recency) with an alphabetical default for predictable navigation.
- LiteLLM-backed prompt execution with automatic history logging and a GUI result pane for reviewing and copying model output.
- Result pane includes a Render Output button that previews LLM responses as rendered Markdown in a dedicated window, alongside a Copy to Text Window action that sends the raw response back into the workspace editor for follow-up editing.
- Prompt engineering workflow that analyses stored prompts against the meta-guidelines and proposes a refined prompt body directly in the editor.
- Prompt dialog can analyse a prompt body to suggest reusable usage scenarios, which are stored with each prompt and rendered in the detail pane.
- Task templates bundle prompts with starter input and surface in the GUI for quick apply/edit/delete workflows.
- Toolbar Info dialog links to the project's author profile and displays the current system's CPU/architecture details alongside the open source licensing summary.
- Prompt list context menu includes Duplicate, Execute, Copy, and description shortcuts so stored prompts can be cloned or run without opening the full editor.

## Getting Started

1. Create a virtual environment and install the application's dependencies in one step:

   ```bash
   pip install -r requirements.txt
   ```

   Dependencies are pinned to maintain compatibility with ChromaDB (for example `numpy<2`). No
   additional extras or package installation steps are required.

2. Provide configuration via environment variables (prefixed `PROMPT_MANAGER_`) or a JSON file referenced through `PROMPT_MANAGER_CONFIG_JSON`. Supported keys include:
   - `DATABASE_PATH` – absolute/relative path to the SQLite database (default `data/prompt_manager.db`).
   - `CHROMA_PATH` – directory for ChromaDB persistence (default `data/chromadb`).
   - `REDIS_DSN` – optional Redis connection string e.g. `redis://localhost:6379/0`.
   - `CACHE_TTL_SECONDS` – positive integer TTL for cached prompts (default `300`).
   - `LITELLM_MODEL` – optional LiteLLM model identifier used for name generation and execution.
   - `EMBEDDING_BACKEND` – one of `deterministic`, `litellm`, or `sentence-transformers` (default `deterministic`).
   - `EMBEDDING_MODEL` – embedding model identifier used by LiteLLM or sentence-transformers backends.
   - `EMBEDDING_DEVICE` – optional device string (e.g. `cpu`, `cuda`) for local sentence-transformers models.
   - `LITELLM_API_VERSION` / `AZURE_OPENAI_API_VERSION` – optional API version (required for Azure OpenAI deployments).

   Copy `config/config.template.json` to `config/config.json` as a starting point for non-secret defaults. LiteLLM API keys are **never** read from JSON; set `PROMPT_MANAGER_LITELLM_API_KEY` (or `AZURE_OPENAI_API_KEY`) in your environment or secrets manager instead.

   When a JSON file is provided, environment variables take precedence. Validation issues raise `config.SettingsError` with actionable context.

3. Build the manager with validated settings:

   ```python
   from config import load_settings
   from core import build_prompt_manager

   settings = load_settings()
   prompt_manager = build_prompt_manager(settings)
   ```

   Supplying a Redis DSN automatically initialises a `redis.Redis` client; omit it to disable caching.

4. Ensure SQLite and ChromaDB have filesystem access at the configured paths.

5. (Optional) Install developer tooling and run the automated tests:

   ```bash
   pip install pytest pytest-cov
   pytest
   ```

6. (Optional) Run static type checks with mypy in strict mode for core modules:

   ```bash
   mypy
   ```

   The configuration in `mypy.ini` enforces strict typing for `core`, `config`, and `models` while allowing tests to stay flexible. Please add annotations and narrow types when extending these packages.

Further modules (session history, execution pipeline) will be introduced in subsequent milestones in line with the blueprint.

## Running the GUI

- Launch the desktop interface. The GUI opens by default:

  ```bash
  python -m main
  ```

- To run in CLI-only bootstrap mode without starting the window:

  ```bash
  python -m main --no-gui
  ```

- Smoke test: run `python -m main --no-gui --print-settings` to validate configuration and backend connectivity without launching the UI; the summary includes path health checks and masks any LiteLLM API key.

- The window exposes a searchable prompt list (left), detail view (right), and toolbar actions for create/edit/delete/refresh backed by the shared `PromptManager` service.
- The prompt detail pane is scrollable so long prompt bodies stay readable and the window remains stable when maximised on Wayland.
- The prompt editor now includes a **Refine** button that runs the prompt engineering meta-prompt and replaces the body with an improved version plus a summary of changes.
- When saving execution results you can apply a 1–10 rating; the average updates the prompt's quality score so filters and recommendations stay aligned with real-world usage.

- Use the workspace beneath the toolbar to paste text, run **Detect Need**, request **Suggest Prompt**, and copy the top-ranked prompt directly to the clipboard.
- As you type, the workspace auto-detects the language (Python, PowerShell, Bash, Markdown, JSON, YAML, or plain text) and applies lightweight syntax highlighting so prompts stay readable.
- Press `Ctrl+K` (or `Ctrl+Shift+P`) to open the command palette for quick actions, or use dedicated shortcuts (`Ctrl+1`…`Ctrl+4`) to jump straight to explain/fix/document/enhance workflows.
- Click the exit icon in the toolbar (or press `Ctrl+Q`) to close the application gracefully and flush background services.
- Tailor the palette in **Settings → Quick actions** by pasting a JSON array of additional actions (identifier, title, description, optional `category_hint`, `tag_hints`, `template`, `prompt_id`, `shortcut`). `prompt_id` can reference a prompt by UUID or name; custom entries override defaults when identifiers collide.
- Task template controls sit above the workspace: pick an existing template to pre-fill the query area and surface linked prompts, or use **New**, **Edit**, **Delete**, and **Clear** to manage templates without leaving the app.

- Use the category, tag, and minimum-quality filters above the list to narrow results as your catalogue grows.

- Dialogs validate required fields and surface backend errors with actionable messaging while keeping data in sync across SQLite, Redis, and ChromaDB.
- When adding prompts the dialog can generate a name from the context field via the **Generate** button. Configure LiteLLM (model + API key) so this suggestion comes from your chosen LLM; otherwise the UI falls back to a heuristic title.
- The **Prompt Body** field is where you paste the actual prompt text that will be sent to the model—it's displayed prominently in the detail pane.
- Use the **Settings** button to adjust LiteLLM routing options (model/base/version), streaming preferences, and quick actions; these values persist to `config/config.json` while API keys entered in the dialog stay in-memory only. Set `PROMPT_MANAGER_LITELLM_API_KEY` in your environment to reuse the key across sessions.
- Intent detection appears beneath the workspace and search bar once you start typing or analysing text, highlighting the inferred category (Debugging, Refactoring, Documentation, etc.), language hints, and the top-matching prompts.
- When LiteLLM is configured, leaving the **Name** and **Description** fields blank will automatically populate them from the prompt body, speeding up catalogue entry for new prompts.
- Use the **Import** button to bring prompts in from JSON files or directories—each import shows a diff preview before changes are applied.
- Use the **Export** button to write the current catalogue to JSON or YAML for backups or sharing.
- The **History** button opens a read-only view of recent prompt executions, including request/response excerpts, durations, and any captured errors.

## Executing Prompts

- LiteLLM support is bundled with the main requirements, so once dependencies are installed the execution backend is ready to use on every platform.
- Configure LiteLLM credentials via the Settings dialog and environment variables: choose the model/Base/Version in Settings and export `PROMPT_MANAGER_LITELLM_API_KEY` (or `AZURE_OPENAI_API_KEY`) so the key comes from your shell or secret store. The GUI never writes the key to disk.
- Paste code or free-form context into the workspace, select a prompt, and click **Run Prompt**. The result pane now exposes **Output** and **Diff** tabs so you can inspect the raw model response or view a unified diff against your original input; **Save Result** captures the run in history with optional notes, and **Copy Result** copies the generated text to the clipboard.
- Continue the conversation after a response with **Continue Chat**, review the live transcript in the new **Chat** tab, and end the session with **End Chat**—the entire dialogue is persisted to history entries.
- Every run is persisted to the new `prompt_executions` table with request excerpts, model responses, duration, status, error details (when present), and LiteLLM token usage metadata.
- Programmatic consumers can call `PromptManager.list_recent_executions()` or `PromptManager.list_executions_for_prompt(prompt_id)` to surface history in future dashboards or integrations.
- Failures are logged with status `failed` and surfaced in the GUI. Successful runs increment `usage_count` for the prompt so catalogue analytics remain accurate.
- Use the **History** tab to filter by status/prompt, search across notes, edit saved entries, and export the current view to CSV—changes flow straight into SQLite via the history APIs.

## CLI Utilities

Prompt Manager ships CLI helpers alongside the GUI:

- `python -m main catalog-export <path>` writes the current repository to JSON or YAML (auto-detected from the extension or forced with `--format`). A YAML export requires `PyYAML`, which is bundled in `requirements.txt`.
- `python -m main suggest "your query"` runs the configured semantic retrieval stack and prints the top matching prompts with detected intent details—useful for validating LiteLLM or sentence-transformer embeddings.
- `python -m main usage-report` summarises the anonymised GUI workspace analytics (counts per action, top intents, top recommended prompts). Pass `--path` to point at a different JSONL log file.

These commands reuse the same validation logic as the GUI; pass an explicit path each time you export.

## Prompt Catalogue

- Prompt Manager no longer seeds default prompts. Add entries through the GUI (**Add** button) or import them via the **Import** button when you have JSON files prepared.
- Use the **Import** button or your own tooling to populate prompts, and the **Export** button or the `catalog-export` CLI command to back up prompts once you are happy with the catalogue.
- Minimum JSON structure:

  ```json
  {
    "name": "Code Review Sentinel",
    "description": "Perform a layered static review on backend code.",
    "category": "Code Analysis",
    "tags": ["code-review", "static-analysis"],
    "quality_score": 9.2,
    "context": "Paste service modules or scripts that require validation.",
    "example_input": "Review this Python module …",
    "example_output": "Highlights issues and provides remediation guidance."
  }
  ```

- Optional fields (`language`, `related_prompts`, `created_at`, `last_modified`, `usage_count`, `source`, and extension slots) map directly to the `Prompt` dataclass. Missing values fall back to sensible defaults and invalid entries are logged during import.
- Populate `category`, `tags`, and `quality_score` to get the most value from the GUI filters and semantic search.
- For LLM-based prompt naming, set `PROMPT_MANAGER_LITELLM_MODEL` and `PROMPT_MANAGER_LITELLM_API_KEY`. The Generate button uses LiteLLM when invoked; exports simply serialise stored prompts.

## Maintenance & Reset

- The **Maintenance** dialog exposes a new **Data Reset** tab with one-click actions to clear the prompt database, wipe the ChromaDB embedding store, or reset all application data (including usage logs) while leaving settings unchanged.
- Each destructive action includes a confirmation prompt and logs the outcome inside the dialog so operators can audit what was removed.

## Telemetry

- ChromaDB anonymized telemetry is disabled by default. Prompt Manager initialises the Chroma client with `anonymized_telemetry=False` to avoid sending usage data and to reduce noisy PostHog-related logs in restricted environments.
- To opt in, set the environment variable `PROMPT_MANAGER_CHROMA_TELEMETRY=1` (feature placeholder). Until a dedicated toggle is exposed in settings, advanced users can fork the project and change the `ChromaSettings` flag in `core/prompt_manager.py:88`.

## Usage Analytics

- Interactions with the GUI intent workspace (Detect Need, Suggest Prompt, Copy Prompt) are logged to `data/logs/intent_usage.jsonl` as anonymised JSON lines. Each entry captures timestamps, query length/hash, detected labels, and the top prompt names—no raw query text is persisted.
- Disable logging by editing `gui/usage_logger.IntentUsageLogger` initialisation or clearing the log file if analytics are not required.
 
## Configuration Precedence

- Precedence (highest → lowest):
  - Explicit overrides passed to `load_settings` / in-memory application changes.
- Application settings file (copy `config/config.template.json` to `config/config.json`, or provide an alternate path via `PROMPT_MANAGER_CONFIG_JSON`).
  - Environment variables (`PROMPT_MANAGER_*`, plus Azure aliases such as `AZURE_OPENAI_API_KEY`).
  - Built-in defaults.

- Naming note: environment variable keys use upper snake case with the `PROMPT_MANAGER_` prefix (e.g., `PROMPT_MANAGER_DATABASE_PATH`). JSON keys use lower snake case (e.g., `database_path`). Env vars always override JSON values when both are provided.

- Example JSON (`config/config.template.json`) provided in-repo as a template:

  ```json
  {
    "database_path": "data/prompt_manager.db",
    "chroma_path": "data/chromadb",
    "redis_dsn": "redis://localhost:6379/0",
    "cache_ttl_seconds": 600,
    "litellm_model": "azure/gpt-4o-mini",
    "litellm_inference_model": null,
    "litellm_stream": false,
    "litellm_drop_params": ["max_tokens", "max_output_tokens", "temperature", "timeout"],
    "litellm_reasoning_effort": null,
    "embedding_model": "text-embedding-3-large"
  }
  ```

- Prompt Manager strips any `litellm_drop_params` entries from outgoing LiteLLM requests locally, ensuring providers such as Azure OpenAI never receive unsupported fields like `drop_params`.
- Configure `litellm_reasoning_effort` with `minimal`, `medium`, or `high` to enable OpenAI reasoning behaviour on `gpt-4.1`, `o3`, or `gpt-5` model families.
- Set `litellm_stream` to `true` to request streaming chat completions; streaming deltas are forwarded to registered callbacks while histories persist the fully aggregated response.
- Provide both a fast LiteLLM model (`PROMPT_MANAGER_LITELLM_MODEL`) and an optional inference model (`PROMPT_MANAGER_LITELLM_INFERENCE_MODEL`) to separate low-latency interactions from slower high-quality generations.
- Supplying `embedding_model` automatically switches the embedding backend to LiteLLM so semantic sync runs against your chosen OpenAI embedding model.

- Environment variables (override JSON when set):

  | Env var | Meaning | Example |
  | --- | --- | --- |
  | `PROMPT_MANAGER_DATABASE_PATH` | SQLite DB path | `/abs/path/prompt_manager.db` |
  | `PROMPT_MANAGER_CHROMA_PATH` | ChromaDB directory | `~/pm/chroma` |
  | `PROMPT_MANAGER_REDIS_DSN` | Redis connection string | `redis://localhost:6379/1` |
  | `PROMPT_MANAGER_CACHE_TTL_SECONDS` | Cache TTL in seconds (>0) | `300` |
  | `PROMPT_MANAGER_CONFIG_JSON` | Path to JSON config used as base | `config/config.json` |
  | `PROMPT_MANAGER_LITELLM_MODEL` | LiteLLM model used for name generation | `gpt-4o-mini` |
  | `PROMPT_MANAGER_LITELLM_INFERENCE_MODEL` | LiteLLM inference model for slower, higher-quality tasks | `gpt-4.1` |
  | `PROMPT_MANAGER_LITELLM_API_KEY` | LiteLLM API key (environment only) | `sk-…` |
  | `PROMPT_MANAGER_LITELLM_API_BASE` | Optional LiteLLM API base override | `https://proxy.example.com` |
  | `PROMPT_MANAGER_LITELLM_DROP_PARAMS` | Comma/JSON list of LiteLLM parameters to drop before sending requests | `max_tokens,temperature` |
  | `PROMPT_MANAGER_LITELLM_REASONING_EFFORT` | Reasoning effort level for OpenAI reasoning models (`minimal`, `medium`, `high`) | `medium` |
  | `PROMPT_MANAGER_LITELLM_STREAM` | Enable streaming chat completions (`true`/`false`) | `true` |
  | `PROMPT_MANAGER_EMBEDDING_MODEL` | Embedding model identifier; enables LiteLLM embeddings automatically | `text-embedding-3-large` |
  | `AZURE_OPENAI_API_KEY` | Alias for LiteLLM API key (Azure) | `azure-key` |
  | `AZURE_OPENAI_ENDPOINT` | Alias for LiteLLM API base (Azure) | `https://<resource>.openai.azure.com` |
  | `PROMPT_MANAGER_EMBEDDING_BACKEND` | Embedding backend (`deterministic`, `litellm`, `sentence-transformers`) | `sentence-transformers` |
  | `PROMPT_MANAGER_EMBEDDING_MODEL` | Embedding model identifier (required when backend ≠ deterministic) | `all-MiniLM-L6-v2` |
  | `PROMPT_MANAGER_EMBEDDING_DEVICE` | Device override for local embedding models | `cuda` |

Notes:
- Paths expand `~` and are resolved to absolute paths.
- Invalid values raise `config.SettingsError` with details.
- When `EMBEDDING_BACKEND` is left at `deterministic`, Prompt Manager uses a reproducible hash-based embedding. Set the backend to `litellm` or `sentence-transformers` with an `EMBEDDING_MODEL` to enable high-quality semantic search. If `EMBEDDING_BACKEND=litellm` and no embedding model is provided, the value from `PROMPT_MANAGER_LITELLM_MODEL` is reused automatically.
