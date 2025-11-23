# PromptManager

PromptManager is a desktop-first prompt operations hub that catalogues, searches, and executes AI prompts with a PySide6 GUI, SQLite persistence, Redis caching, and ChromaDB-backed semantic retrieval. It targets Python 3.12+ and ships with strict typing, automated testing, and reproducible configuration baked in.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Python 3.12 or newer is required (3.13 when available).
- Dependencies are pinned in `requirements.txt`; no extras are needed for the base app.

## Quick Start

```bash
# 1. Configure paths and providers (override as needed)
export PROMPT_MANAGER_DATABASE_PATH="data/prompt_manager.db"
export PROMPT_MANAGER_CHROMA_PATH="data/chromadb"
export PROMPT_MANAGER_LITELLM_MODEL="gpt-4o-mini"
export PROMPT_MANAGER_LITELLM_API_KEY="sk-***"

# 2. Validate configuration and services
python -m main --no-gui --print-settings

# 3. Launch the GUI
python -m main
```

- Copy `config/config.template.json` to `config/config.json` to persist non-secret defaults; environment variables always win.
- Omit `PROMPT_MANAGER_LITELLM_API_KEY` to run without prompt execution; deterministic embeddings remain available for offline workflows.

## Features

- Semantic search that embeds every prompt facet (name, description, tags, scenarios) via LiteLLM or sentence-transformers and returns cosine-ranked matches directly from ChromaDB.
- Full PySide6 GUI with list/search/detail panes, in-place CRUD, fork/version history, quick action palette, and refined prompt editing workflows.
- Prompt execution workspace with LiteLLM-backed runs, streaming output, chat-style transcripts, rating-based quality feedback, and export/import utilities.
- Typed configuration loader (`config.settings`) that validates paths, TTLs, and provider settings from env vars or JSON, failing fast with actionable errors.
- Redis-backed caching plus optional deterministic embeddings for air-gapped deployments; maintenance dialog exposes one-click reset tooling.
- CLI helpers (`catalog-export`, `suggest`, `usage-report`, `reembed`) reuse the same validation stack for automation-friendly integrations.

## Developer

See the full contributor guide in [`README-DEV.md`](README-DEV.md) for development environment setup, environment variable matrix, testing/type-checking workflow, embedding and GUI deep dives, and maintenance procedures.

## License

PromptManager is licensed under the [MIT License](LICENSE.md).

## Updates

<<<<<<< HEAD
Updates:
  v0.17.1 - 2025-11-23 - Restructured README and introduced README-DEV developer handbook.
=======
## CLI Utilities

Prompt Manager ships CLI helpers alongside the GUI:

- `python -m main catalog-export <path>` writes the current repository to JSON or YAML (auto-detected from the extension or forced with `--format`). A YAML export requires `PyYAML`, which is bundled in `requirements.txt`.
- `python -m main suggest "your query"` runs the configured semantic retrieval stack and prints the top matching prompts with detected intent details—useful for validating LiteLLM or sentence-transformer embeddings.
- `python -m main usage-report` summarises the anonymised GUI workspace analytics (counts per action, top intents, top recommended prompts). Pass `--path` to point at a different JSONL log file.
- `python -m main reembed` deletes the current ChromaDB directory and regenerates embeddings for every stored prompt—run this after changing embedding backends or when the vector store is corrupted.

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
- Seed additional categories via `PROMPT_MANAGER_CATEGORIES_PATH` (path to a JSON file containing category objects) or inline JSON in `PROMPT_MANAGER_CATEGORIES`; the new **Manage** dialog can still add or rename categories at runtime and persists them to SQLite.
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
    "litellm_workflow_models": {
      "prompt_execution": "inference",
      "prompt_structure_refinement": "inference"
    },
    "embedding_backend": "litellm",
    "embedding_model": "text-embedding-3-large"
  }
  ```

- Prompt Manager strips any `litellm_drop_params` entries from outgoing LiteLLM requests locally, ensuring providers such as Azure OpenAI never receive unsupported fields like `drop_params`.
- Configure `litellm_reasoning_effort` with `minimal`, `medium`, or `high` to enable OpenAI reasoning behaviour on `gpt-4.1`, `o3`, or `gpt-5` model families.
- Set `litellm_stream` to `true` to request streaming chat completions; streaming deltas are forwarded to registered callbacks while histories persist the fully aggregated response.
- Provide both a fast LiteLLM model (`PROMPT_MANAGER_LITELLM_MODEL`) and an optional inference model (`PROMPT_MANAGER_LITELLM_INFERENCE_MODEL`) to separate low-latency interactions from slower high-quality generations.
- Use the settings dialog's **Routing** tab (or `PROMPT_MANAGER_LITELLM_WORKFLOW_MODELS`) to assign each workflow to the fast or inference model without editing code.
- The settings dialog now includes a dedicated LiteLLM **Embedding model** field so you can control which embedding endpoint powers semantic search (defaults to `text-embedding-3-large`).

- Environment variables (override JSON when set):

  | Env var | Meaning | Example |
  | --- | --- | --- |
  | `PROMPT_MANAGER_DATABASE_PATH` | SQLite DB path | `/abs/path/prompt_manager.db` |
  | `PROMPT_MANAGER_CHROMA_PATH` | ChromaDB directory | `~/pm/chroma` |
  | `PROMPT_MANAGER_REDIS_DSN` | Redis connection string | `redis://localhost:6379/1` |
  | `PROMPT_MANAGER_CACHE_TTL_SECONDS` | Cache TTL in seconds (>0) | `300` |
  | `PROMPT_MANAGER_CONFIG_JSON` | Path to JSON config used as base | `config/config.json` |
  | `PROMPT_MANAGER_CATEGORIES_PATH` | Path to a JSON file containing category definitions | `config/categories.json` |
  | `PROMPT_MANAGER_CATEGORIES` | Inline JSON array of category definitions | `[{"slug": "review", "label": "Review"}]` |
  | `PROMPT_MANAGER_LITELLM_MODEL` | LiteLLM model used for name generation | `gpt-4o-mini` |
  | `PROMPT_MANAGER_LITELLM_INFERENCE_MODEL` | LiteLLM inference model for slower, higher-quality tasks | `gpt-4.1` |
  | `PROMPT_MANAGER_LITELLM_WORKFLOW_MODELS` | JSON map of workflows to `fast`/`inference` (e.g. `{"prompt_execution": "inference", "prompt_structure_refinement": "inference"}`) | `{"prompt_execution": "inference"}` |
  | `PROMPT_MANAGER_LITELLM_API_KEY` | LiteLLM API key (environment only) | `sk-…` |
  | `PROMPT_MANAGER_LITELLM_API_BASE` | Optional LiteLLM API base override | `https://proxy.example.com` |
  | `PROMPT_MANAGER_LITELLM_DROP_PARAMS` | Comma/JSON list of LiteLLM parameters to drop before sending requests | `max_tokens,temperature` |
  | `PROMPT_MANAGER_LITELLM_REASONING_EFFORT` | Reasoning effort level for OpenAI reasoning models (`minimal`, `medium`, `high`) | `medium` |
  | `PROMPT_MANAGER_LITELLM_STREAM` | Enable streaming chat completions (`true`/`false`) | `true` |
  | `PROMPT_MANAGER_EMBEDDING_MODEL` | LiteLLM embedding model identifier used for semantic search | `text-embedding-3-large` |
  | `AZURE_OPENAI_API_KEY` | Alias for LiteLLM API key (Azure) | `azure-key` |
  | `AZURE_OPENAI_ENDPOINT` | Alias for LiteLLM API base (Azure) | `https://<resource>.openai.azure.com` |
  | `PROMPT_MANAGER_EMBEDDING_BACKEND` | Embedding backend (`litellm`, `sentence-transformers`, `deterministic`) | `sentence-transformers` |
  | `PROMPT_MANAGER_EMBEDDING_DEVICE` | Device override for local embedding models | `cuda` |

Notes:
- Paths expand `~` and are resolved to absolute paths.
- Invalid values raise `config.SettingsError` with details.
- Prompt Manager now defaults to LiteLLM embeddings to keep semantic search aligned with production quality. Override `EMBEDDING_BACKEND=deterministic` if you need the legacy hash-based vectors, or switch to `sentence-transformers` with `PROMPT_MANAGER_EMBEDDING_MODEL` + `PROMPT_MANAGER_EMBEDDING_DEVICE` for local embeddings. When using LiteLLM, the settings dialog exposes a dedicated **Embedding model** field so you can point at any provider-supported embedding endpoint.
>>>>>>> 141a5ac9635b0b1981065e94fc333123265045ee
