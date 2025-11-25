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
- Live Jinja2 template preview pane beside the workspace that renders prompts with JSON variables, ships custom filters (`truncate`, `slugify`, `json`), and validates payloads with optional JSON Schema or Pydantic feedback in real time.
- Response Style registry with dedicated GUI tab, CRUD, clipboard/export helpers, and reusable tone/formatting presets that can be attached to executions for consistent voice.
- Prompt execution workspace with LiteLLM-backed runs, streaming output, chat-style transcripts, rating-based quality feedback, export/import utilities, and persisted execution history complete with context metadata (model, streaming flag, style hints).
- Typed configuration loader (`config.settings`) that validates paths, TTLs, and provider settings from env vars or JSON, failing fast with actionable errors.
- Redis-backed caching plus optional deterministic embeddings for air-gapped deployments; maintenance dialog exposes one-click reset tooling.
- CLI helpers (`catalog-export`, `suggest`, `usage-report`, `reembed`) reuse the same validation stack for automation-friendly integrations.

### Response Styles & Formatting

Use the **Response Styles** tab to capture reusable tone, voice, structure, and formatting rules. Each style stores:

- Human-readable name/description for discovery.
- Tone/voice hints plus free-form guidelines.
- Rich format instructions and illustrative examples.

Styles can be copied, exported to Markdown, or deleted from the tab. They are persisted in SQLite and referenced from execution metadata so downstream automation (or future runtime hooks) can apply the same tone and structure without duplicating large instruction blocks.

### Execution History & Auditing

Every run logs to SQLite with request/response snippets, latency, token usage, status, and structured context metadata (prompt info, model, streaming flag, response-style fingerprints). The GUI History tab and programmatic APIs (`list_recent_executions`, `list_executions_for_prompt`) surface the data for reviews, and the log payload is ready for downstream analytics or incident investigations.

## Developer

See the full contributor guide in [`README-DEV.md`](README-DEV.md) for development environment setup, environment variable matrix, testing/type-checking workflow, embedding and GUI deep dives, and maintenance procedures.

## License

PromptManager is licensed under the [MIT License](LICENSE.md).
