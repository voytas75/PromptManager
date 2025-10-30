# Prompt Manager

Prompt Manager is a desktop-focused application for cataloguing, searching, and executing AI prompts. The project follows the architecture captured in `PromptManagerBlueprint.txt` and targets PySide6 for the GUI while relying on SQLite for canonical storage, ChromaDB for semantic retrieval, and Redis for responsive caching.

## Current Status

- Core data model defined in `models/prompt_model.py` with an extensible schema (`ext1`–`ext5`) and serialization helpers.
- `PromptRepository` (`core/repository.py`) persists prompts to SQLite and exposes CRUD operations.
- `PromptManager` service (`core/prompt_manager.py`) coordinates SQLite persistence, Redis-backed caching, and ChromaDB semantic search.
- Typed configuration loader in `config/settings.py` validates paths/TTL values and can hydrate from environment variables or a JSON file.
- Project scaffolding prepared for config, GUI, data, and test layers.

## Getting Started

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Dependencies are pinned to maintain compatibility with ChromaDB (for example `numpy<2`).

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

Further modules (GUI, session history, execution pipeline) will be introduced in subsequent milestones in line with the blueprint.

## Configuration Precedence

- Precedence (highest → lowest):
  - Environment variables (`PROMPT_MANAGER_*`)
  - JSON file pointed by `PROMPT_MANAGER_CONFIG_JSON`
  - Built-in defaults

- Example JSON (`config/config.json`):

  ```json
  {
    "database_path": "data/prompt_manager.db",
    "chroma_path": "data/chromadb",
    "redis_dsn": "redis://localhost:6379/0",
    "cache_ttl_seconds": 600
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
