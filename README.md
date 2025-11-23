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
