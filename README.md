# PromptManager

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![Quality Gates](https://github.com/voytas75/PromptManager/actions/workflows/quality-gates.yml/badge.svg)](https://github.com/voytas75/PromptManager/actions/workflows/quality-gates.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](README.md#project-status)

PromptManager is a local-first desktop app for **capturing, organizing, retrieving, and refining prompt assets** in one place.

Its product center is simple: act as a **canonical home for prompt assets** so useful prompts and LLM queries do not stay scattered across chats, notes, scripts, markdown files, and ad-hoc experiments.

It combines a PySide6 GUI, SQLite persistence, semantic search, prompt editing, and optional lightweight execution support, but the product center stays the prompt catalog itself rather than analytics, chains, or general AI-workstation behavior.

<p align="center">
  <img src="docs/images/main.png" alt="Prompt catalogue and workspace view" width="45%">
  <img src="docs/images/template.png" alt="Template preview validating JSON variables" width="45%">
</p>

## Who it is for

PromptManager is designed for people who actively work with prompts and want a more structured local-first workflow:

- **AI developers** who maintain reusable prompts across projects
- **prompt engineers** who want a durable prompt base instead of scattered snippets
- **solo builders** who want one local prompt workspace instead of loose Markdown files and chat fragments
- **operators and researchers** who collect prompts from many places and want to find and reuse them quickly

## What it helps with

Use PromptManager when you want to:

- capture useful prompts or LLM queries into a searchable local catalog
- normalize drafts into reusable prompt assets with better titles, metadata, and provenance
- reopen recent work quickly and inspect prompt context without hunting through metadata
- preview and validate templated prompts before running them
- execute prompts from a dedicated workspace when lightweight validation or reuse helps
- review prompt lineage, reuse context, and supporting execution history when needed
- keep prompt work local-first, with optional external providers

## Core capabilities

- **Prompt catalog** — store, search, tag, edit, fork, and organize prompts as durable assets
- **Quick Capture + Draft Promote** — capture raw prompt/query text fast, inspect it as a draft, then promote it into a reusable prompt asset with bounded title-quality improvement for weak draft titles
- **Recent reopen + inspection cues** — get back to recent prompts quickly and see draft/source/last-modified context without opening raw metadata
- **Quick reuse actions** — copy the real prompt body from detail view with one obvious `Copy Prompt` action, or open the prompt in the workspace without auto-running it
- **Semantic retrieval** — find prompts by meaning, not only exact text
- **Template preview** — render Jinja2 templates with JSON variables and validation feedback
- **Lightweight execution workspace** — run prompts with LiteLLM-backed models and keep execution history when validation or reuse benefits from it
- **Supporting prompt parts and analytics** — reuse supporting prompt components and inspect lightweight history signals without changing the product center

## Quick start

### 1. Create a virtual environment

Use either the existing `pip` flow or `uv`.

**Option A — pip + venv**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

**Option B — uv**

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Optional: add your API key

Start by copying the example environment file:

```bash
cp .env.example .env
```

If you want to execute prompts with LiteLLM-backed providers, fill in the LiteLLM values in `.env` or export them directly:

```bash
export PROMPT_MANAGER_LITELLM_MODEL="gpt-4o-mini"
export PROMPT_MANAGER_LITELLM_API_KEY="sk-***"
```

If you skip this step, PromptManager can still be used for local cataloguing, editing, and offline workflows.

### 3. Validate configuration

```bash
python -m main --no-gui --print-settings
# or
uv run python -m main --no-gui --print-settings
```

### 4. Launch the app

```bash
python -m main
# or
uv run python -m main
```

## Minimal local setup

PromptManager works best as a **local-first desktop tool**:

- SQLite stores your prompt catalog and execution history
- ChromaDB powers semantic search
- Redis is optional
- LiteLLM is optional for prompt execution
- web search integrations are optional

That means you can start with a local catalog first, then add providers only when you need execution or retrieval enhancements.

## Optional integrations

PromptManager supports optional integrations for:

- **LiteLLM** for prompt execution
- **Tavily / Exa / Serper / SerpApi / Google Programmable Search** for web search enrichment
- **Redis** for caching

Advanced environment variables and full configuration details are documented in:
- [`docs/README-DEV.md`](docs/README-DEV.md)
- [`docs/web_search_plan.md`](docs/web_search_plan.md)

The repository root also includes [`.env.example`](.env.example) as a safe starting point for local configuration.

## Why PromptManager

Most prompt workflows start as some combination of:
- Markdown files
- text snippets
- chat history
- copied JSON payloads
- half-reusable templates

That works for a while, but it breaks down once you want repeatability, reuse, better retrieval, or proper versioning.

PromptManager gives you a dedicated local-first home for prompt assets:
- capture-oriented
- structured
- searchable
- reuse-friendly
- execution-aware when needed

## Project status

PromptManager is currently in **beta** and under active development.

The current focus is:
- strengthening the core prompt asset loop: capture, normalize, retrieve, inspect, reuse, refine
- improving low-friction draft capture, promotion, and reuse workflows
- refining template and validation ergonomics where they directly support prompt assets
- keeping execution, analytics, chains, sharing, and voice as supporting surfaces rather than the product center
- keeping the desktop experience responsive and local-first

## Detailed features by product role

### Core prompt asset workflow

- Full PySide6 GUI with list/search/detail panes, in-place CRUD, fork/version history, quick action palette, and refined prompt editing workflows.
- Quick Capture, Draft Promote, recent reopen, and inspection cues keep rough prompt/query material close to the catalog instead of leaving it scattered across chats, notes, and ad-hoc files.
- Semantic search embeds prompt facets such as name, description, tags, and scenarios via LiteLLM or sentence-transformers and returns cosine-ranked matches from ChromaDB.
- Live Jinja2 template preview renders prompts with JSON variables, includes custom filters (`truncate`, `slugify`, `json`), and validates payloads with optional JSON Schema or Pydantic feedback.
- Prompt Parts provide reusable response styles, system instructions, redaction rules, and formatting presets so supporting prompt components can stay structured and reusable.

### Supporting reuse, validation, and local operations

- Prompt execution workspace supports LiteLLM-backed runs, streaming output, chat-style transcripts, export/import utilities, and persisted execution history when validation or reuse benefits from it.
- Dedicated Prompt Template editor exposes LiteLLM system prompts with inline validation and reset-to-default controls.
- Typed configuration loading (`config.settings`) validates paths, TTLs, and provider settings from env vars or JSON.
- Optional Redis-backed caching plus deterministic embeddings keep the app useful in offline or air-gapped workflows.
- Maintenance tooling includes one-click reset actions and a snapshot export that bundles the SQLite database, Chroma store, and manifest before risky changes.
- CLI helpers such as `catalog-export`, `suggest`, `usage-report`, and `reembed` reuse the same validation stack for automation-friendly workflows.

### Supporting history and light analytics

- Every run is logged to SQLite with request/response snippets, latency, token usage, status, and structured context metadata. The GUI History tab and programmatic APIs (`list_recent_executions`, `list_executions_for_prompt`) surface that data for review, troubleshooting, and light curation.
- Token usage tracking shows per-run prompt/completion/total tokens, running session totals, and dashboard summaries. Expanded rollups are outlined in `docs/token_usage_plan.md`.
- Analytics dashboard surfaces usage frequency, cost breakdowns, benchmark success, intent success trends, and embedding health with CSV export.
- Execution analytics CLI (`history-analytics`) summarises success rates, latency, and rating trends.
- Diagnostics CLI targets (`python -m main diagnostics embeddings|analytics`) verify embedding backends, scan Chroma for inconsistencies, and export deeper analysis data.

### Secondary surfaces, available but not the product center

- Enhanced Prompt Workbench offers a guided wizard, block palette, template variable dialogs, Template Preview integration, LiteLLM Brainstorm/Peek/Run controls, and one-click export.
- Web search enrichment supports Exa, Tavily, Serper, SerpApi, and Google Programmable Search, with a workspace toggle to inject live snippets before execution.
- Prompt chaining pipelines are available via CLI (`prompt-chain-*`) and the dedicated **Chain** tab in the GUI.
- Prompt sharing supports ShareText, Rentry, and PrivateBin, with clipboard-friendly share URLs plus returned delete/edit credentials when available.
- Voice playback can stream LiteLLM text-to-speech audio as it downloads; toggle via `PROMPT_MANAGER_LITELLM_TTS_STREAM`.
- Workspace appearance controls let you adjust font family, size, and colour for prompt output and chat panes at runtime.
- Benchmark CLI (`benchmark`) runs prompts across configured LiteLLM models and prints duration and token metrics.

These surfaces can still be useful, but they are supporting or secondary to the core job: keeping prompt assets captured, understandable, retrievable, reusable, and refinable.

## Developer

See the full contributor guide in [`README-DEV.md`](docs/README-DEV.md) for development environment setup, environment variable matrix, testing/type-checking workflow, embedding and GUI deep dives, and maintenance procedures.
The staged web search rollout plan (covering Exa, Tavily, Serper, SerpApi, and Google Programmable Search) lives in [`docs/web_search_plan.md`](docs/web_search_plan.md).

## Changelog

Track release highlights and historical updates in [`CHANGELOG.md`](docs/CHANGELOG.md).

## License

PromptManager is licensed under the [MIT License](LICENSE.md).
