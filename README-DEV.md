# PromptManager – Developer Guide

PromptManager is a PySide6 desktop application for managing reusable AI prompts with SQLite persistence, Redis caching, and ChromaDB-powered semantic retrieval. This document captures the engineering conventions, environment expectations, and deep technical workflows required to extend the project safely.

## Updates

Updates:
  v0.20.0 - 2025-11-29 - Documented the Enhanced Prompt Workbench launch path and tooling stack.
  v0.17.4 - 2025-11-23 - Added Similar context menu action to surface embedding-based recommendations plus body-size and rating sort orders to spotlight verbose or high-quality templates inside the GUI.
  v0.17.1 - 2025-11-23 - Added developer handbook aligned with AGENTS.md directives.

## Current Status

- `models/prompt_model.Prompt` holds the canonical schema (extension slots `ext1`–`ext5`, serialization helpers, embedding document builder).
- `core.repository.PromptRepository` provides SQLite CRUD and backs the GUI plus CLI utilities.
- `core.prompt_manager.PromptManager` orchestrates persistence, Redis caching, LiteLLM execution, and ChromaDB similarity queries.
- PySide6 GUI (`main.py --gui`) exposes list/search/detail panes, prompt editor with refinement workflow, quick action palette, notes, history, and taxonomy management dialogs.
- Notification center, status tracking for long-running embedding/LLM tasks, and preference profile ensure responsive UX.

## Toolchain & Quality Gates

- **Python**: 3.12+ (3.13 when stable) with `pyright` in strict mode; annotations are mandatory (no `type: ignore` in `core/`).
- **Formatting/Linting**: `ruff --fix` plus `black --line-length 88`. Import ordering follows ruff/isort (builtin → stdlib → third-party → local).
- **Testing**: `pytest -n auto --cov=core --cov-fail-under=90` with `pytest-asyncio`, `pytest-cov`, and `hypothesis` for parsing/generation code. Mock all external HTTP/DB calls (`respx`, `vcrpy`, `pytest-mock`).
- **Automation**: `nox -s format lint typecheck test` runs the full CI-equivalent workflow. Add new sessions as needed but keep parity with AGENTS.md.
- **Security & Resilience**: wrap external I/O in timeouts, provide custom exception hierarchy, never use bare `except`, and include actionable context plus retries with exponential backoff where transient failures may occur.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional developer extras
pip install ruff black pytest pyright nox
nox -s all
```

There is no dedicated `requirements-dev.txt`; install the lint/type/test toolchain manually as shown above.

1. Copy `config/config.template.json` to `config/config.json` for non-secret defaults.
2. Export environment variables for anything secret or machine-specific (see below). Environment variables override JSON, which overrides built-in defaults.
3. Run `python -m main --no-gui --print-settings` to verify filesystem paths, Redis, LiteLLM, and ChromaDB connectivity before coding.

## Configuration & Environment Variables

All settings are defined via `pydantic-settings` in `config/settings.py`. Provide values through environment variables (preferred) or JSON. Key variables:

| Variable | Description | Example |
| --- | --- | --- |
| `PROMPT_MANAGER_DATABASE_PATH` | SQLite database path | `data/prompt_manager.db` |
| `PROMPT_MANAGER_CHROMA_PATH` | ChromaDB persistence directory | `data/chromadb` |
| `PROMPT_MANAGER_REDIS_DSN` | Redis connection string | `redis://localhost:6379/0` |
| `PROMPT_MANAGER_CACHE_TTL_SECONDS` | Cache TTL in seconds (>0) | `600` |
| `PROMPT_MANAGER_CONFIG_JSON` | Path to base JSON config | `config/config.json` |
| `PROMPT_MANAGER_CATEGORIES_PATH` / `PROMPT_MANAGER_CATEGORIES` | Category seed definitions (file or inline JSON) | `[{"slug": "review","label": "Review"}]` |
| `PROMPT_MANAGER_LITELLM_MODEL` | LiteLLM model for prompt execution/name generation | `gpt-4o-mini` |
| `PROMPT_MANAGER_LITELLM_INFERENCE_MODEL` | High-quality workflow model | `gpt-4.1` |
| `PROMPT_MANAGER_LITELLM_WORKFLOW_MODELS` | JSON mapping of workflows to `fast`/`inference` | `{"prompt_execution": "inference"}` |
| `PROMPT_MANAGER_LITELLM_API_KEY` / `AZURE_OPENAI_API_KEY` | LiteLLM or Azure credentials (environment only) | `sk-***` |
| `PROMPT_MANAGER_LITELLM_API_BASE` / `AZURE_OPENAI_ENDPOINT` | Override LiteLLM base URL | `https://proxy.example.com` |
| `PROMPT_MANAGER_LITELLM_DROP_PARAMS` | Comma/JSON list of parameters stripped before sending | `max_tokens,temperature` |
| `PROMPT_MANAGER_LITELLM_REASONING_EFFORT` | `minimal`, `medium`, or `high` for OpenAI reasoning models | `medium` |
| `PROMPT_MANAGER_LITELLM_STREAM` | Enable streaming responses (`true`/`false`) | `true` |
| `PROMPT_MANAGER_EMBEDDING_BACKEND` | `litellm`, `sentence-transformers`, or `deterministic` | `sentence-transformers` |
| `PROMPT_MANAGER_EMBEDDING_MODEL` | Embedding model identifier | `text-embedding-3-large` |
| `PROMPT_MANAGER_EMBEDDING_DEVICE` | Device hint for local embeddings | `cuda` |
| `PROMPT_MANAGER_CHROMA_TELEMETRY` | Opt-in flag for Chroma telemetry (`1` to enable) | `0` |

Secrets must never be committed; rely on `.env` files ignored by git or host-level secret stores.

## Detailed Getting Started

1. **Validate configuration**
   ```bash
   python -m main --no-gui --print-settings
   ```
   This command checks JSON + env precedence, path writability, Redis connectivity (if configured), and masks API keys.

2. **Initialize databases**
   - Ensure `data/` is writable; SQLite and ChromaDB directories are created automatically.
   - Run `python -m main reembed` whenever you change embedding backends to rebuild vectors consistently.

3. **Seed prompts (optional)**
   - Use `python -m main catalog-export data/catalog.json` to snapshot the current library.
   - Import via the GUI (**Import** button) or custom tooling that writes JSON matching `Prompt` fields.

## Embedding & Search Behaviour

- Every prompt's searchable document concatenates: name, description, category, tags, context, example input/output, and stored scenarios.
- Embeddings are produced via the configured backend:
  - `litellm`: delegates to provider embeddings (default `text-embedding-3-large`).
  - `sentence-transformers`: runs locally; set `PROMPT_MANAGER_EMBEDDING_DEVICE` for GPU.
  - `deterministic`: offline hashing for smoke tests.
- Search queries embed the entire user phrase and ask ChromaDB for nearest neighbours; results are already cosine-ranked and displayed as-is in the GUI.
- The GUI shows similarity scores (`[0.91]`) when search is active; the sort dropdown is disabled to preserve ranking integrity.

## Running the GUI

- Launch normally:
  ```bash
  python -m main
  ```
- Launch without the GUI (bootstrap/CLI mode):
  ```bash
  python -m main --no-gui
  ```
- Smoke-test config and dependencies:
  ```bash
  python -m main --no-gui --print-settings
  ```

Key UI capabilities:
- List/search/detail panes with CRUD operations, diff viewer, fork lineage, and scroll-safe prompt bodies.
- Workspace under the toolbar supports Detect Need, Suggest Prompt, Copy Prompt flows, language auto-detection, and quick clearing.
- Enhanced Prompt Workbench (🆕 toolbar button) launches a modal surface with a guided wizard, block palette, Template Preview integration, LiteLLM Brainstorm/Peek/Run Once helpers, variable dialogs, and export-to-repository wiring so teams can iterate on drafts without touching the main catalogue view.
- A Template Preview frame below the workspace renders the selected prompt as a strict Jinja2 template, accepts JSON variables, and surfaces validation/missing-field issues instantly.
- Command palette (`Ctrl+K` / `Ctrl+Shift+P`) and shortcuts (`Ctrl+1`–`Ctrl+4`) jump directly into explain/fix/document/enhance workflows.
- Category/tag/quality filters plus taxonomy manager keep catalogues organized.
- Settings dialog controls LiteLLM routing, streaming, quick actions, and embedding configuration; API keys entered in the GUI stay in memory only.
- Share panel (bordered row beneath the action buttons) lets users pick what to publish (body-only, body + description, or body + description + scenarios), toggle metadata inclusion, and then pick a provider (ShareText today). Uploads happen on a worker thread, and the resulting URL is copied to the clipboard automatically.

### Template Preview & Validation

- Variable input must be a JSON object; malformed payloads disable rendering with inline errors.
- Optional schema textarea supports JSON Schema (Draft 2020-12 via `jsonschema`) or derived Pydantic models. Choose the mode from the combo box to highlight failing top-level fields and show descriptive error text.
- Custom filters available in prompt bodies: `truncate` (adds ellipsis), `slugify` (matches category helpers), and `json` (pretty printing with optional indent parameter).
- Missing variables, schema violations, and undefined placeholders paint statuses red/orange in the variable list and block the rendered preview.
- Dependencies `Jinja2==3.1.4` and `jsonschema==4.23.0` ship in `requirements.txt`; no optional extras are required for developers.

## Executing Prompts

- Configure LiteLLM credentials via environment variables; the GUI never writes keys to disk.
- Running a prompt logs executions to the `prompt_executions` table with durations, statuses, token usage, errors, and snippets.
- Continue conversations via **Continue Chat**; transcripts appear in the **Chat** tab and are persisted.
- Save results with notes and optional 1–10 ratings; averages feed into quality filters.
- Programmatic access is available through `PromptManager.list_recent_executions()` and `PromptManager.list_executions_for_prompt(prompt_id)`.

Every log entry also stores structured context metadata (prompt snapshot, executor model, streaming flag, request/response character counts, and optional prompt-part fingerprints). Inspect the metadata via the GUI history detail pane or fetch it directly from `PromptExecution.metadata` for downstream analytics.

## Prompt Parts Workflow

- Capture reusable prompt segments (response styles, system instructions, output formatters, evaluation rubrics) from the **Prompt Parts** tab. The dialog records name, prompt part classification, description, tone, voice, format instructions, guidelines, tags, and illustrative examples; timestamps and versions are maintained automatically.
- Entries are persisted in the `response_styles` table (now with a `prompt_part` column) and surfaced through `PromptManager.list_response_styles` (with `include_inactive` and `search` filters) alongside CRUD helpers.
- GUI actions provide copy-to-clipboard, Markdown preview/export, and duplication capabilities so content writers can curate libraries without digging into SQLite.
- When executions capture a prompt part, the metadata stores the part ID plus flattened instructions so downstream automations can reuse the formatting contract.

## CLI Utilities

| Command | Purpose |
| --- | --- |
| `python -m main catalog-export <path> [--format json|yaml]` | Export prompts; YAML requires PyYAML (already bundled). |
| `python -m main suggest "search query"` | Run semantic retrieval and print top matches with intent metadata. |
| `python -m main usage-report [--path <file>]` | Summarize anonymized GUI analytics (counts, intents, recommendations). |
| `python -m main history-analytics [--window-days N --limit M --trend-window K]` | Display execution success rates, durations, and ratings for recent prompts. |
| `python -m main reembed` | Rebuild the ChromaDB vector store after backend/model changes or corruption. |
| `python -m main benchmark --prompt <uuid> [--model <id>] --request "…"` | Execute one or more prompts across configured LiteLLM models and compare duration/token usage alongside history stats. |
| `python -m main refresh-scenarios <uuid> [--max-scenarios N]` | Regenerate and persist scenario lists for a prompt via LiteLLM or the heuristic fallback. |

These commands share the same validation logic as the GUI; pass explicit paths as needed.

## Testing & Type Checking

- **Unit/Integration Tests**: `pytest -n auto --cov=core --cov-fail-under=90`.
- **Property-Based Tests**: Use `hypothesis` for prompt parsing, token-length sensitive logic, and JSON import/export features.
- **Type Checking**: `pyright` must pass with zero warnings; enable strict mode for new packages and keep `pyproject.toml` aligned.
- **Static Analysis**: `ruff --fix` (lint + autofix) followed by `black --check .`.
- **Recommended workflow**:
  ```bash
  nox -s format lint typecheck test
  ```

## Maintenance, Telemetry & Analytics

- **Maintenance dialog**: Provides buttons to clear SQLite prompts, wipe ChromaDB embeddings, or reset all application data (usage logs, cache) with confirmation prompts and logging.
- **Snapshot backups**: Use the **Create Backup Snapshot** button in the maintenance dialog to zip the SQLite database, Chroma persistence directory, and a JSON manifest before running destructive tasks; the archive path is user-selected so it can be stored outside the project tree.
- **Category health panel**: Review per-category prompt counts, active prompt ratios, recent execution timestamps, and success rates directly inside the maintenance dialog; use the Refresh button after batch edits.
- **Telemetry**: ChromaDB anonymized telemetry is disabled (`anonymized_telemetry=False`). Set `PROMPT_MANAGER_CHROMA_TELEMETRY=1` to opt in or adjust `core/prompt_manager.py` if you need different defaults.
- **Usage analytics**: GUI intent workspace interactions are logged to `data/logs/intent_usage.jsonl` (timestamp, hashed query metadata, detected intents, top prompts). Disable via `gui.usage_logger.IntentUsageLogger` instantiation or by clearing the log path.
- **Analytics dashboard**: The GUI **Analytics** tab (and `python -m main diagnostics analytics`) pulls execution history, benchmark metadata, embeddings health, and intent usage logs into configurable charts. Set the window/prompt limits via the panel controls or CLI flags (`--window-days`, `--prompt-limit`), choose datasets (`usage`, `model_costs`, `benchmark`, `intent`, `embedding`), and export any dataset with `--export-csv` or the tab's **Export CSV** button for downstream BI tooling.

## Prompt Catalogue Management

- No default prompts are seeded; import via GUI or CLI.
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
- Optional fields (`language`, `related_prompts`, `created_at`, `last_modified`, `usage_count`, `source`, extensions) map directly to the dataclass attributes; invalid values raise `config.SettingsError` or are logged during import.
- Category management relies on the `CategoryRegistry`. Seed defaults with `PROMPT_MANAGER_CATEGORIES_PATH` or inline JSON (`PROMPT_MANAGER_CATEGORIES`), then use the GUI **Manage** dialog (or `PromptManager.create_category` / `update_category`) to add, rename, or archive entries. Each prompt stores both a user-facing label and a slug so renames propagate without breaking history or filters, and archives keep historical prompts readable without appearing in filters.
