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
- **solo builders** who want one local prompt catalog instead of loose Markdown files and chat fragments
- **operators and researchers** who collect prompts from many places and want to find and reuse them quickly

## What it helps with

Use PromptManager when you want to:

- capture useful prompts or LLM queries into a searchable local catalog
- normalize drafts into reusable prompt assets with better titles, metadata, and provenance
- reopen recent work quickly and inspect prompt context without hunting through metadata
- preview templated prompts before reuse or optional validation
- use a lightweight workspace only when prompt validation or reuse benefits from it
- review prompt lineage, reuse context, and light supporting history when needed
- keep prompt work local-first, with optional external providers

## Recommended usage path

If you are unsure where to start, use this sequence:

1. **Quick Capture** a useful prompt or LLM query before it gets lost in chat, notes, or scratch files.
2. **Promote Draft** once it is worth keeping as a reusable prompt asset.
3. **Use Recent or search** to get back to it quickly.
4. **Inspect the detail view** to confirm fit, context, provenance, and any lineage cues.
5. **Reuse with `Copy Prompt` or `Open in Workspace`** when you want the stored prompt body or lightweight validation.

Short version:

**`Quick Capture` → `Promote Draft` → `Recent` / search → inspect → `Copy Prompt` or `Open in Workspace`**

This is the canonical PromptManager v1 flow. It keeps the catalog at the center and treats execution, analytics, chains, sharing, and other secondary surfaces as supporting tools rather than the default front door.

For a slightly fuller operator-facing version, see [`docs/canonical-usage-path-v1.md`](docs/canonical-usage-path-v1.md).

## Core capabilities

- **Prompt catalog** — store, search, tag, edit, fork, and organize prompts as durable assets
- **Quick Capture + Draft Promote** — capture raw prompt/query text fast, inspect it as a draft, then promote it into a reusable prompt asset with bounded title-quality improvement for weak draft titles
- **Recent reopen + inspection cues** — get back to recent prompts quickly and see draft/source/last-modified context without opening raw metadata
- **Quick reuse actions** — copy the real prompt body from detail view with one obvious `Copy Prompt` action, or open the prompt in the workspace without auto-running it
- **Semantic retrieval** — find prompts by meaning, not only exact text
- **Template preview** — render Jinja2 templates with JSON variables and validation feedback
- **Optional validation workspace** — run prompts with LiteLLM-backed models only when lightweight validation or reuse benefits from it
- **Supporting prompt parts and light history signals** — reuse supporting prompt components and inspect secondary support surfaces without changing the product center

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
- validation-aware when needed

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

### Supporting validation and local operations

These capabilities support the prompt catalog when validation, maintenance, or automation becomes useful, but they are not the default front door:

- an optional LiteLLM-backed execution workspace for lightweight validation or reuse
- Prompt Template editing and template-preview support
- typed configuration loading plus optional Redis-backed caching
- maintenance tooling such as reset actions and snapshot export
- CLI helpers such as `catalog-export`, `suggest`, `usage-report`, and `reembed`

### Supporting history and light analytics

These surfaces exist to help curation and troubleshooting, not to redefine the product:

- persisted run history in SQLite, surfaced in the GUI History tab and related APIs
- lightweight token, latency, and rating visibility
- `history-analytics` and diagnostics helpers for bounded operator insight

### Secondary surfaces, available but intentionally demoted

The repository still includes broader surfaces that may be useful in some workflows, but they are not the product center and should be read as optional:

- Prompt Workbench and related guided authoring helpers
- web search enrichment before execution
- prompt chaining flows
- prompt sharing helpers
- voice/TTS playback
- workspace appearance controls
- benchmark tooling

The core job remains the same: keep prompt assets captured, understandable, retrievable, reusable, and refinable.

## Developer

See the full contributor guide in [`README-DEV.md`](docs/README-DEV.md) for development environment setup, environment variable matrix, testing/type-checking workflow, embedding and GUI deep dives, and maintenance procedures.
The staged web search rollout plan (covering Exa, Tavily, Serper, SerpApi, and Google Programmable Search) lives in [`docs/web_search_plan.md`](docs/web_search_plan.md).

## Changelog

Track release highlights and historical updates in [`CHANGELOG.md`](docs/CHANGELOG.md).

## License

PromptManager is licensed under the [MIT License](LICENSE.md).
