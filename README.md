# Prompt Manager

Prompt Manager is a desktop-focused application for cataloguing, searching, and executing AI prompts. The project follows the architecture captured in `PromptManagerBlueprint.txt` and targets PySide6 for the GUI while relying on SQLite for canonical storage, ChromaDB for semantic retrieval, and Redis for responsive caching.

## Current Status

- Core data model defined in `models/prompt_model.py` with an extensible schema (`ext1`–`ext5`) and serialization helpers.
- `PromptRepository` (`core/repository.py`) persists prompts to SQLite and exposes CRUD operations.
- `PromptManager` service (`core/prompt_manager.py`) coordinates SQLite persistence, Redis-backed caching, and ChromaDB semantic search.
- Typed configuration loader in `config/settings.py` validates paths/TTL values and can hydrate from environment variables or a JSON file.
- Initial PySide6 GUI accessible via `--gui`, offering list/search/detail panes with create/edit/delete dialogs.

## Getting Started

1. Create a virtual environment and install core dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Dependencies are pinned to maintain compatibility with ChromaDB (for example `numpy<2`).

   Alternatively, install the project as a package:

   ```bash
   pip install .
   ```

   To enable the desktop interface install the optional GUI extras:

   ```bash
   pip install -r requirements-gui.txt
   # or, when installing the package form:
   pip install .[gui]
   ```

2. Provide configuration via environment variables (prefixed `PROMPT_MANAGER_`) or a JSON file referenced through `PROMPT_MANAGER_CONFIG_JSON`. Supported keys include:
   - `DATABASE_PATH` – absolute/relative path to the SQLite database (default `data/prompt_manager.db`).
   - `CHROMA_PATH` – directory for ChromaDB persistence (default `data/chromadb`).
   - `REDIS_DSN` – optional Redis connection string e.g. `redis://localhost:6379/0`.
   - `CACHE_TTL_SECONDS` – positive integer TTL for cached prompts (default `300`).

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

5. Run the automated tests to validate storage integration:

   ```bash
   pytest
   ```

6. (Optional) Run static type checks with mypy in strict mode for core modules:

   ```bash
   mypy
   ```

   The configuration in `mypy.ini` enforces strict typing for `core`, `config`, and `models` while allowing tests to stay flexible. Please add annotations and narrow types when extending these packages.

Further modules (session history, execution pipeline) will be introduced in subsequent milestones in line with the blueprint.

## Running the GUI

- Ensure the optional GUI requirements are installed (`pip install .[gui]` or `pip install -r requirements-gui.txt`).

- Launch the desktop interface. The GUI opens by default:

  ```bash
  python -m main
  ```

- To run in CLI-only bootstrap mode without starting the window:

  ```bash
  python -m main --no-gui
  ```

- Smoke test: run `python -m main --no-gui --print-settings` to validate configuration and backend connectivity without launching the UI.

- The window exposes a searchable prompt list (left), detail view (right), and toolbar actions for create/edit/delete/refresh backed by the shared `PromptManager` service.

- Use the category, tag, and minimum-quality filters above the list to narrow results as your catalogue grows.

- Dialogs validate required fields and surface backend errors with actionable messaging while keeping data in sync across SQLite, Redis, and ChromaDB.

## Prompt Catalogue

- On startup Prompt Manager imports the packaged catalogue in `catalog/prompts.json`, seeding six high-signal prompts across analysis, refactoring, debugging, documentation, reporting, and enhancement.
- Override the catalogue by setting `PROMPT_MANAGER_CATALOG_PATH` (or `catalog_path` in `config/config.json`) to a JSON file or to a directory containing JSON files. Entries with matching names update existing prompts; new names are inserted automatically.
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

## Telemetry

- ChromaDB anonymized telemetry is disabled by default. Prompt Manager initialises the Chroma client with `anonymized_telemetry=False` to avoid sending usage data and to reduce noisy PostHog-related logs in restricted environments.
- To opt in, set the environment variable `PROMPT_MANAGER_CHROMA_TELEMETRY=1` (feature placeholder). Until a dedicated toggle is exposed in settings, advanced users can fork the project and change the `ChromaSettings` flag in `core/prompt_manager.py:88`.


## Configuration Precedence

- Precedence (highest → lowest):
  - Environment variables (`PROMPT_MANAGER_*`)
  - JSON file pointed by `PROMPT_MANAGER_CONFIG_JSON`
  - Built-in defaults

- Naming note: environment variable keys use upper snake case with the `PROMPT_MANAGER_` prefix (e.g., `PROMPT_MANAGER_DATABASE_PATH`). JSON keys use lower snake case (e.g., `database_path`). Env vars always override JSON values when both are provided.

- Example JSON (`config/config.json`) provided in-repo as a template:

  ```json
  {
    "database_path": "data/prompt_manager.db",
    "chroma_path": "data/chromadb",
    "redis_dsn": "redis://localhost:6379/0",
    "cache_ttl_seconds": 600,
    "catalog_path": "catalog/prompts.json"
  }
  ```

- Environment variables (override JSON when set):

  | Env var | Meaning | Example |
  | --- | --- | --- |
  | `PROMPT_MANAGER_DATABASE_PATH` | SQLite DB path | `/abs/path/prompt_manager.db` |
  | `PROMPT_MANAGER_CHROMA_PATH` | ChromaDB directory | `~/pm/chroma` |
  | `PROMPT_MANAGER_REDIS_DSN` | Redis connection string | `redis://localhost:6379/1` |
  | `PROMPT_MANAGER_CACHE_TTL_SECONDS` | Cache TTL in seconds (>0) | `300` |
  | `PROMPT_MANAGER_CONFIG_JSON` | Path to JSON config used as base | `config/config.json` |

Notes:
- Paths expand `~` and are resolved to absolute paths.
- Invalid values raise `config.SettingsError` with details.
