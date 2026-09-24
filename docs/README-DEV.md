# PromptManager – Developer Guide

PromptManager is a PySide6 desktop application for managing reusable AI prompts with SQLite persistence, optional Redis caching, and ChromaDB-powered semantic retrieval. This is the deep technical/operational reference; [`README-DEV.md`](../README-DEV.md) is the short onboarding entry point. This document does not supersede the product SSOT or the active near-term plan.

## Current Status

- `models/prompt_model.Prompt` holds the canonical schema (extension slots `ext1`–`ext5`, serialization helpers, embedding document builder).
- `core.repository.PromptRepository` provides SQLite CRUD and backs the GUI plus CLI utilities.
- `core.prompt_manager.PromptManager` orchestrates persistence, optional Redis caching, LiteLLM execution, and ChromaDB similarity queries.
- PySide6 GUI (`main.py --gui`) exposes list/search/detail panes, prompt editor with refinement workflow, quick action palette, notes, history, and taxonomy management dialogs.
- Notification center, status tracking for long-running embedding/LLM tasks, and preference profile ensure responsive UX.
- Product SSOT now lives in [`docs/product-ssot.md`](product-ssot.md) and should be treated as the canonical source for what PromptManager is, what is core, and what is secondary/later.
- Active near-term planning lives in [`docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`](plans/2026-05-10-product-direction-ssot-next-cycle.md), and delivered slice/history tracking lives in [`docs/STATUS.md`](STATUS.md).

## Toolchain & Quality Gates

- **Python**: 3.13+. The GitHub Quality Gates workflow runs `pyright main.py config models` as its type-check step; branch-protection/required-check settings are separate and must be verified before saying it blocks merges. The full configured strict scope (`main.py`, `core/`, `config/`, `gui/`, `models/`, and `tests/`) was verified locally for the historical 0.23.1 release closeout. Use full strict Pyright locally for broad changes where appropriate, but do not call it exact CI parity unless the workflow is deliberately expanded. Annotations remain mandatory for new or touched code (no `type: ignore` in `core/`).
- **Formatting/Linting**: `ruff check --fix .` followed by `ruff format .` (line length 100). Import ordering follows ruff/isort (builtin → stdlib → third-party → local).
- **Testing**: `pytest -n auto --cov=core --cov-report=term-missing --cov-fail-under=80` with `pytest-asyncio`, `pytest-cov`, and `hypothesis` for parsing/generation code. Under `uv`, prefer `uv sync --extra dev` first, then `uv run pytest ...`, or use one-shot `uv run --extra dev pytest ...`. Mock all external HTTP/DB calls (`respx`, `vcrpy`, `pytest-mock`).
- **Automation**: If you separately install `nox` (not included in the `dev` extra), `nox -s format lint typecheck test` is a broader local quality run: its Pyright session checks the full configured scope. The GitHub workflow is `.github/workflows/quality-gates.yml`, including `pyright main.py config models`; Nox is not exact CI parity.
- **Security & Resilience**: wrap external I/O in timeouts, provide custom exception hierarchy, never use bare `except`, and include actionable context plus retries with exponential backoff where transient failures may occur.

### What moved out of AGENTS.md

The repository-level `AGENTS.md` is intentionally short and operational. Keep long-form project policy here instead of growing `AGENTS.md` back into a handbook.

High-value conventions that still apply:
- preserve existing docstring and file-history conventions when editing files that already use them,
- keep release-gate language aligned with the *actually enforced* CI scope,
- prefer bounded slice verification over pretending that every change must solve all historical debt,
- keep product/UI policy and deep workflow notes in `docs/` rather than in the repo-level agent contract.

### Quality policy: release gate vs slice verification vs debt scans

Treat quality checks as three separate layers:

1. **GitHub workflow checks (should stay green)**
   - `.github/workflows/quality-gates.yml` runs on push and pull requests; branch protection / required-check status must be checked in GitHub separately.
   - Current workflow scope is:
     - `ruff check --fix .`
     - `ruff format .`
     - `ruff check .` + `ruff format --check .`
     - `pyright main.py config models`
     - `pytest -n auto --cov=core --cov-report=term-missing --cov-fail-under=80`
     - `git diff --exit-code`
   - Do not describe full-repo strict Pyright as a required gate unless CI really enforces it.

2. **Bounded slice verification (required for local implementation work)**
   - For each feature/fix slice, run the smallest credible proof for the touched area before commit.
   - Minimum expectation:
     - targeted pytest for the touched feature/module,
     - file-level or narrow-directory Pyright for touched typed files,
     - full pytest when the slice touches shared runtime flows or broad integration seams.
   - Goal: prove the local slice is safe without pretending all historical debt is part of the same decision.

3. **Debt scans / expansion probes (required only when explicitly promoted)**
   - Commands such as full `pyright`, `pyright gui --stats`, or `pyright tests --stats` are debt-radar tools by default.
   - For the completed 0.23.1 release closeout, the full configured strict Pyright scan was required and passed locally with zero errors; this is historical verification, not a pending release action.
   - The scan remains distinct from CI parity until `.github/workflows/quality-gates.yml` is deliberately expanded.

Operational rule: when touching a file that already carries strict-typing debt, do not widen the debt casually; prefer making the touched file flatter or at least not measurably worse.

### Release hardening baseline (Beta → stable)

- Keep the repository's Ruff, CI-scope Pyright, and pytest+coverage checks green for delivery; fail-fast settings validation and resilient external I/O remain engineering expectations. Whether checks block merges depends on GitHub branch protection, not on the workflow file alone.
- CI uses `.github/workflows/quality-gates.yml` with `pip install -e .[dev]`; run its individual `.venv/bin/ruff`, `.venv/bin/pyright main.py config models`, and `.venv/bin/pytest ...` invocations for comparable checks. `uv sync --extra dev` installs from the tracked lockfile and is useful for reproducibility, but is **not** exact dependency-install parity with CI. `nox -s all` requires separately installed Nox and runs a broader full-Pyright check.
- Validate settings early during development with `python -m main --no-gui --print-settings` or `python scripts/validate_settings.py`; configuration failures must be actionable and stop execution.
- For HTTP I/O, prefer `httpx.AsyncClient(timeout=...)` plus retry helpers (e.g., `core.retry.async_retry`) and mock calls in tests with `httpx.MockTransport`, `respx`, or `vcrpy` (no live external calls in CI).

## Environment Setup

Use either the existing `pip` flow or `uv`.

**Option A — pip + venv**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]

# Optional Nox command wrapper (separate from the project's dev extra)
python -m pip install nox
nox -s all
```

**Option B — uv (recommended)**

```bash
uv sync --extra dev
uv run pyright
uv run pytest -n auto --cov=core --cov-report=term-missing --cov-fail-under=80
```

`pyproject.toml` is the single source of truth for project dependencies. For `uv`, prefer `uv sync --extra dev` as the canonical repo setup path. If you only want a one-shot test invocation on a fresh machine, use:

```bash
uv run --extra dev pytest -n auto --cov=core --cov-report=term-missing --cov-fail-under=80
```

Do not assume plain `uv run pytest` will work in a freshly created environment, because `pytest` lives in the optional `dev` extra rather than in the base runtime dependencies. `uv sync --extra dev` uses `uv.lock`; CI installs via pip instead, so the two dependency resolutions are not identical by contract. Nox is also outside the dev extra; install it separately before using the Nox commands below.

1. Copy `.env.example` to `.env` for a safe local starting point.
2. If you copy `config/config.template.json` to `config/config.json` for non-secret defaults, check its `embedding_backend` before indexing or search: the template selects `litellm` and, as JSON outranks `.env`, overrides `.env.example`'s `deterministic` choice. For provider-free indexing/search set `"embedding_backend": "deterministic"` **in that JSON** (or leave the template uncopied and confirm the effective setting). Do not infer provider-free behavior from `.env` alone after copying the template.
3. Export environment variables for secrets or machine-specific settings (see below). For overlapping application settings, explicit constructor overrides take precedence, then JSON configuration, then canonical environment variables, then provider aliases, then file secrets. A value in JSON can therefore override a corresponding environment variable; inspect the effective source before assuming an env change took effect.
   - Set `PROMPT_MANAGER_ENV_FILE` if you want to load a different dotenv path.
4. Run `python -m main --no-gui --print-settings` (or `uv run python -m main --no-gui --print-settings`) to inspect validated effective settings and reported path state before coding. This command does not initialize the manager or prove path writability or Redis, LiteLLM, or ChromaDB connectivity. Use the bounded, provider-free `doctor` subcommands for local inspections; live backend probes require explicit `--live` and may incur cost.

## Configuration & Environment Variables

All settings are defined via `pydantic-settings` in `config/settings.py`. Use JSON for overlapping application settings (it takes precedence over environment variables) and environment variables for secrets or values not set in JSON. Key variables:

| Variable | Description | Example |
| --- | --- | --- |
| `PROMPT_MANAGER_DATABASE_PATH` | SQLite database path | `data/prompt_manager.db` |
| `PROMPT_MANAGER_CHROMA_PATH` | ChromaDB persistence directory | `data/chromadb` |
| `PROMPT_MANAGER_REDIS_DSN` | Redis connection string (leave unset to disable caching) | `redis://localhost:6379/0` |
| `PROMPT_MANAGER_CACHE_TTL_SECONDS` | Cache TTL in seconds (>0) | `600` |
| `PROMPT_MANAGER_CONFIG_JSON` | Path to base JSON config | `config/config.json` |
| `PROMPT_MANAGER_CATEGORIES_PATH` / `PROMPT_MANAGER_CATEGORIES` | Category seed definitions (file or inline JSON) | `[{"slug": "review","label": "Review"}]` |
| `PROMPT_MANAGER_LITELLM_MODEL` | LiteLLM model for prompt execution/name generation | `gpt-4o-mini` |
| `PROMPT_MANAGER_LITELLM_INFERENCE_MODEL` | High-quality workflow model | `gpt-4.1` |
| `PROMPT_MANAGER_LITELLM_WORKFLOW_MODELS` | JSON mapping of workflows to `fast`/`inference` | `{"prompt_execution": "inference"}` |
| `PROMPT_MANAGER_LITELLM_API_KEY` / `AZURE_OPENAI_API_KEY` | LiteLLM or Azure credentials (environment only) | `sk-***` |
| `PROMPT_MANAGER_LITELLM_API_BASE` / `AZURE_OPENAI_ENDPOINT` | Override LiteLLM base URL | `https://proxy.example.com` |
| `PROMPT_MANAGER_LITELLM_DROP_PARAMS` | Comma-separated string or JSON array of parameters stripped before sending | `max_tokens,temperature` or `["max_tokens","temperature"]` |
| `PROMPT_MANAGER_LITELLM_REASONING_EFFORT` | `minimal`, `medium`, or `high` for OpenAI reasoning models | `medium` |
| `PROMPT_MANAGER_LITELLM_STREAM` | Enable streaming responses (`true`/`false`) | `true` |
| `PROMPT_MANAGER_LITELLM_LOGGING` | Allow LiteLLM library logs to surface (`true`/`false`) | `false` |
| `PROMPT_MANAGER_LITELLM_TTS_MODEL` | LiteLLM text-to-speech model id used for voice playback | `openai/tts-1` |
| `PROMPT_MANAGER_LITELLM_TTS_STREAM` | Stream LiteLLM TTS audio so playback starts mid-download (`true`/`false`) | `true` |
| `PROMPT_MANAGER_WEB_SEARCH_PROVIDER` | Web search provider slug (`exa`, `tavily`, `serper`, `serpapi`, `google`, `random`, or leave empty to disable) | `tavily` |
| `PROMPT_MANAGER_EXA_API_KEY` / `EXA_API_KEY` | Exa web search API key (environment only) | `exa_***` |
| `PROMPT_MANAGER_TAVILY_API_KEY` / `TAVILY_API_KEY` | Tavily web search API key (environment only) | `tvly-***` |
| `PROMPT_MANAGER_SERPER_API_KEY` / `SERPER_API_KEY` | Serper.dev web search API key (environment only) | `serper-***` |
| `PROMPT_MANAGER_SERPAPI_API_KEY` / `SERPAPI_API_KEY` | SerpApi web search API key (environment only) | `serpapi-***` |
| `PROMPT_MANAGER_GOOGLE_API_KEY` / `GOOGLE_API_KEY` | Google Programmable Search API key (environment only) | `AIza***` |
| `PROMPT_MANAGER_GOOGLE_CSE_ID` / `GOOGLE_CSE_ID` | Google Programmable Search Engine ID (environment only) | `1234567890abcdef:ghijklmnop` |
| `PROMPT_MANAGER_AUTO_OPEN_SHARE_LINKS` | Auto-open shared URLs in the default browser (`true`/`false`) | `true` |
| `PROMPT_MANAGER_PRIVATEBIN_URL` | Base URL for the PrivateBin instance used for sharing (include trailing slash) | `https://privatebin.net/` |
| `PROMPT_MANAGER_PRIVATEBIN_EXPIRATION` | PrivateBin expiration key (`5min`, `1day`, `1week`, `1month`, `1year`, `never`, etc.) | `1week` |
| `PROMPT_MANAGER_PRIVATEBIN_FORMAT` | Formatter advertised to PrivateBin (`plaintext`, `markdown`, or `syntaxhighlighting`) | `markdown` |
| `PROMPT_MANAGER_PRIVATEBIN_COMPRESSION` | Compression method before encryption (`zlib` or `none`) | `zlib` |
| `PROMPT_MANAGER_PRIVATEBIN_BURN_AFTER_READING` | Delete PrivateBin pastes immediately after the first read (`true`/`false`) | `false` |
| `PROMPT_MANAGER_PRIVATEBIN_OPEN_DISCUSSION` | Allow comments/discussion threads on new PrivateBin pastes (`true`/`false`) | `false` |
| `PROMPT_MANAGER_EMBEDDING_BACKEND` | `litellm`, `sentence-transformers` (requires optional package), or `deterministic` | `litellm` (built-in default; `.env.example` selects `deterministic`) |
| `PROMPT_MANAGER_EMBEDDING_MODEL` | Embedding model identifier | `text-embedding-3-large` |
| `PROMPT_MANAGER_EMBEDDING_DEVICE` | Device hint for local embeddings | `cuda` |
| `PROMPT_MANAGER_PROMPT_OUTPUT_FONT_FAMILY` | Workspace output font family | `JetBrains Mono` |
| `PROMPT_MANAGER_PROMPT_OUTPUT_FONT_SIZE` | Workspace output font size (pt) | `13` |
| `PROMPT_MANAGER_PROMPT_OUTPUT_FONT_COLOR` | Workspace output font colour (hex) | `#B5CFED` |
| `PROMPT_MANAGER_CHAT_FONT_FAMILY` | Workspace chat font family | `Segoe UI` |
| `PROMPT_MANAGER_CHAT_FONT_SIZE` | Workspace chat font size (pt) | `12` |
| `PROMPT_MANAGER_CHAT_FONT_COLOR` | Workspace chat font colour (hex) | `#B5CFED` |

Secrets must never be committed; rely on `.env` files ignored by git or host-level secret stores.
The committed [`.env.example`](../.env.example) file is the canonical placeholder template for local setup and must never contain real credentials.

### Logging config vs provider logging

PromptManager startup uses the runtime logging bootstrap in `main.py`. It loads a local, ignored `config/logging.conf` when present; otherwise it loads the tracked `config/logging.conf.example`. Pass `--logging-config <path>` to use an explicit file instead.

Copy the example when you need host-specific verbosity:

```bash
cp config/logging.conf.example config/logging.conf
```

The local file controls the **global/root console logging level** for the whole process. If it sets `root` or the console handler to `DEBUG`, the terminal can fill with debug output from third-party libraries such as ChromaDB, OpenAI, `httpcore`, and `asyncio` during normal GUI startup and shutdown.

This is separate from `PROMPT_MANAGER_LITELLM_LOGGING`.

- `config/logging.conf` controls overall Python logging verbosity.
- `PROMPT_MANAGER_LITELLM_LOGGING` only controls whether LiteLLM-specific library logs are allowed to surface.

So `PROMPT_MANAGER_LITELLM_LOGGING=false` does **not** silence broad debug output when the root logger is already configured at `DEBUG`.

For normal local GUI use, prefer `INFO` or `WARN` in `config/logging.conf`.

Selecting `PROMPT_MANAGER_WEB_SEARCH_PROVIDER="random"` rotates calls between whichever providers currently have API keys configured; if only one provider has a key, Random behaves like that provider until another key is available.

See [`docs/web_search_plan.md`](web_search_plan.md) for the staged web search integration plan (Exa + Tavily + Serper + SerpApi + Google Programmable Search) if you are extending the provider surface.

## Versioning & release trail

Treat versioning as a deliberately small system:

- **`pyproject.toml` is the SSOT for the package version.**
- **`docs/CHANGELOG.md` is the SSOT for release history.**
- **`*.egg-info/` and `PKG-INFO` are generated local metadata, not version SSOT and not a git-tracked source of truth.**

### Version selection (Semantic Versioning)

Use `MAJOR.MINOR.PATCH` for releases:

- Increment **MAJOR** only for a breaking public contract change that requires operator or integrator migration.
- Increment **MINOR** for backward-compatible, user-visible functionality, including a new public CLI command.
- Increment **PATCH** for backward-compatible bug fixes, documentation-only corrections, and internal changes that do not add public functionality.

### Practical rules

- When cutting a new release, select the version under the policy above, bump `project.version` in `pyproject.toml`, and move the current changelog content from `## [Unreleased]` into a dated release heading.
- Leave a fresh empty `## [Unreleased]` section at the top after closing a release.
- Do not manually treat `prompt_manager.egg-info/PKG-INFO` as canonical project state. It reflects the local installed/editable package metadata and may be stale until refreshed.
- If the GUI or maintenance surfaces show an old version, refresh the editable install metadata locally:
  - `pip install -e .[dev]`
  - or `uv pip install -e .[dev]`
- After a version bump, verify the runtime-visible version with:
  - `python - <<'PY'
from importlib.metadata import version
print(version("prompt-manager"))
PY`

### Minimal release checklist

1. Update `pyproject.toml` version.
2. Close the current changelog under a dated release heading.
3. Keep a fresh `## [Unreleased]` section ready for the next cycle.
4. Refresh editable metadata locally if needed.
5. Spot-check that `importlib.metadata.version("prompt-manager")` matches the intended release.

This keeps repo truth and runtime-visible package metadata aligned without pretending generated local metadata should be hand-maintained in git.

## Detailed Getting Started

1. **Validate configuration**
   ```bash
   python -m main --no-gui --print-settings
   ```
   This prints the validated effective settings, source/precedence labels, and reported path state with credentials masked. It does not initialize services, test path writability, or verify Redis, LiteLLM, or ChromaDB connectivity.

2. **Initialize databases**
   - Ensure `data/` is writable; SQLite and ChromaDB directories are created automatically.
   - If you decide to change embedding backends, first stop all writers and make a consistent, restorable backup of SQLite and Chroma data; verify the backup before proceeding. **Create Backup Snapshot** alone is not proof of a consistent SQLite backup while WAL writes are pending (see Maintenance below). Then plan `python -m main reembed` deliberately. This command **resets the Chroma collection before rebuilding**; individual embedding failures may leave a partial index. A configured remote embedding backend can send stored prompt content to a provider and incur cost. Confirm the effective backend and review the command result before relying on search again; do not use it as a routine first-run check.

3. **Seed prompts (optional)**
   - Use `python -m main catalog-export data/catalog.json` to snapshot the current library.
   - Import via the GUI (**Import** button) or custom tooling that writes JSON matching `Prompt` fields.

## Embedding & Search Behaviour

- Every prompt's searchable document concatenates: name, description, category, tags, context, example input/output, and stored scenarios.
- Embeddings are produced via the configured backend:
  - `litellm`: delegates to provider embeddings (default `text-embedding-3-large`).
  - `sentence-transformers`: runs locally when the optional `sentence-transformers` package and model are installed; it is not included in the base or `dev` dependency set. Set `PROMPT_MANAGER_EMBEDDING_DEVICE` for GPU if available.
  - `deterministic`: offline hashing for smoke tests.
- LiteLLM embedding readiness and actual calls are distinct: missing credentials can be reported as a readiness issue, but this does not guarantee deterministic fallback. An available LiteLLM embedding function may still be selected and fail on use; choose `embedding_backend: "deterministic"` explicitly in JSON when provider-free indexing and search are required. A deterministic fallback is used when the LiteLLM embedding function itself is unavailable.
- Search queries embed the entire user phrase and ask ChromaDB for nearest neighbours; results are already cosine-ranked and displayed as-is in the GUI.
- The GUI shows similarity scores (`[0.91]`) when search is active; the sort dropdown is disabled to preserve ranking integrity.
- When a provider (Exa, Tavily, Serper, SerpApi, or Google Programmable Search) is configured, the workspace and Chain tab expose a “Use web search” checkbox (checked by default). Leaving it on runs a provider query (prompt metadata + user input) before execution and prepends available summaries/highlights with source links. In the workspace path only, web context over ~5,000 words may be condensed by the fast LiteLLM model; the Chain path prepends context without that condensation. Unchecking the box (or running `prompt-chain-run --no-web-search`) disables web-search enrichment only. Execution still requires the configured LiteLLM model and can contact its provider and incur cost; this is not an offline or provider-free run.

## Running the GUI

- Launch an installed package:
  ```bash
  prompt-manager
  ```
- Launch from a repository checkout:
  ```bash
  python -m main
  # or
  uv run python -m main
  ```
- Launch without the GUI (bootstrap/CLI mode):
  ```bash
  prompt-manager --no-gui
  # or, from a checkout
  uv run python -m main --no-gui
  ```
- Smoke-test config and dependencies:
  ```bash
  prompt-manager --no-gui --print-settings
  # or, from a checkout
  uv run python -m main --no-gui --print-settings
  ```
- On Linux/WSL, the GUI preflight inspects PySide6's bundled `xcb` plugin before constructing `QApplication`. If it reports unresolved system libraries, install the listed runtime dependencies (Ubuntu/Debian: `sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxkbcommon-x11-0`) and retry. The preflight never invokes `sudo` or mutates the host.

Key UI capabilities:
- List/search/detail panes with CRUD operations, diff viewer, fork lineage, and scroll-safe prompt bodies.
- In the Create/Edit Prompt dialog, a description can be entered manually or requested with **Generate description**. Without a model this shows a local excerpt; with a configured model it prompts before a provider/cost-bearing call. Saving a blank description (including after clearing one in Edit) uses only a local excerpt from the body. A blank name is suggested locally from the complete body at save, not from the first keystroke; explicit **Generate** for a name prompts before a configured model call. CLI `prompt-add` still requires an explicit description and does not run these GUI helpers.
- Workspace under the toolbar supports Detect Need, Suggest Prompt, Copy Prompt flows, language auto-detection, and quick clearing.
- Enhanced Prompt Workbench (🆕 toolbar button) launches a modal surface with a guided wizard, block palette, Template Preview integration, LiteLLM Brainstorm/Peek/Run Once helpers, variable dialogs, and export-to-repository wiring so teams can iterate on drafts without touching the main catalogue view.
- Workspace result metadata surfaces per-run token usage, and a dedicated label keeps running session totals alongside all-time totals fetched from history so authors immediately see spend.
- A historical token rollup proposal lives in [`docs/token_usage_plan.md`](token_usage_plan.md). It is not the active UI/CLI roadmap; select new work from the [near-term product plan](plans/2026-05-10-product-direction-ssot-next-cycle.md) after validating a current operator need.
- The GUI forces Qt's **Fusion** style at startup (see `gui/application.py`) so the palette-driven theming looks identical on Windows/macOS/Linux. If you experiment with alternative styles, verify Guided mode and the Link Variable dialog still use the dark palette before committing.
- Workspace appearance controls (font family/size/colour for output and chat panes) are configurable via settings or env vars and apply at runtime; tooltips for the “Use web search” toggle reflect the active provider (Exa, Tavily, Serper, SerpApi, Google, or Random).
- Guided wizard now uses a custom-styled dialog (not `QWizard`) to avoid native theme overrides; adjust `GuidedPromptWizard` inside `gui/workbench/workbench_window.py` when changing layout, palette, or button flow.
- A Template Preview frame below the workspace renders the selected prompt as a strict Jinja2 template, accepts JSON variables, and surfaces validation/missing-field issues instantly.
- Command palette (`Ctrl+K` / `Ctrl+Shift+P`) and shortcuts (`Ctrl+1`–`Ctrl+4`) jump directly into explain/fix/document/enhance workflows.
- Category/tag/quality filters plus taxonomy manager keep catalogues organized.
- Settings dialog controls LiteLLM routing, streaming, quick actions, embedding configuration, and now the Integrations tab for web search providers; API keys entered in the GUI stay in memory only.
- Share panel (bordered row beneath the action buttons) lets users pick what to publish (body-only, body + description, or body + description + scenarios), toggle metadata inclusion, and then pick a provider: ShareText for quick plain-text pastes, Rentry for markdown pages with editable slugs/edit codes (per the [official CLI/API README](https://github.com/radude/rentry/blob/master/README.md), retrieved 2025-12-07), or PrivateBin for zero-knowledge AES-256-GCM uploads per the [PrivateBin API](https://github.com/PrivateBin/PrivateBin/wiki/API), retrieved 2025-12-07. Uploads happen on a worker thread, the resulting URL is copied to the clipboard automatically, and we surface the delete token or edit code returned by each provider.

### Template Preview & Validation

- Variable input must be a JSON object; malformed payloads disable rendering with inline errors.
- Optional schema textarea supports JSON Schema (Draft 2020-12 via `jsonschema`) or derived Pydantic models. Choose the mode from the combo box to highlight failing top-level fields and show descriptive error text.
- Custom filters available in prompt bodies: `truncate` (adds ellipsis), `slugify` (matches category helpers), and `json` (pretty printing with optional indent parameter).
- Missing variables, schema violations, and undefined placeholders paint statuses red/orange in the variable list and block the rendered preview.
- Dependencies such as `Jinja2` and `jsonschema` are declared in `pyproject.toml`; no extra install step is required beyond `pip install -e .[dev]`.

## Executing Prompts

- Configure LiteLLM credentials via environment variables; the GUI never writes keys to disk.
- Set `PROMPT_MANAGER_LITELLM_TTS_MODEL` to the provider-specific text-to-speech model when you want the workspace to read results aloud; keep it unset to prevent audio requests.
- Leave `PROMPT_MANAGER_LITELLM_TTS_STREAM` at its default (`true`) to start playback while LiteLLM audio downloads, or set it to `false` if your environment requires fully downloaded files before playback.
- The Output overlay includes a speaker icon that streams LiteLLM audio for the latest result; it automatically disables while prompts are streaming or when Qt Multimedia support is missing.
- Running a prompt logs executions to the `prompt_executions` table with durations, statuses, token usage, errors, and snippets.
- The History tab now includes a **Tokens** column plus detail/export fields that show prompt/completion/total numbers for each execution, and the summary footer aggregates the totals shown in the current filter window.
- Continue conversations via **Continue Chat**; transcripts appear in the **Chat** tab and are persisted.
- Save results with notes and optional 1–10 ratings; averages feed into quality filters.
- Interpret the stored prompt `quality_score` as an average of optional saved-run ratings, not an automatically validated or universal prompt-quality measure; inspect `rating_count` before relying on it. Ratings remain a single optional signal, not a roadmap driver. The existing Favorite toggle/filter is a separate personal retrieval choice, not an inferred quality or ranking signal. Policy and revisit trigger: `docs/product-ssot.md`.
- Programmatic access is available through `PromptManager.list_recent_executions()` and `PromptManager.list_executions_for_prompt(prompt_id)`.

Every log entry also stores structured context metadata (prompt snapshot, executor model, streaming flag, request/response character counts, and optional prompt-part fingerprints). Inspect the metadata via the GUI history detail pane or fetch it directly from `PromptExecution.metadata` for downstream analytics.

## Prompt Parts Workflow

- Capture reusable prompt segments (response styles, system instructions, output formatters, evaluation rubrics) from the **Prompt Parts** tab. The dialog records name, prompt part classification, description, tone, voice, format instructions, guidelines, tags, and illustrative examples; timestamps and versions are maintained automatically.
- Entries are persisted in the `response_styles` table (now with a `prompt_part` column) and surfaced through `PromptManager.list_response_styles` (with `include_inactive` and `search` filters) alongside CRUD helpers.
- GUI actions provide copy-to-clipboard, Markdown preview/export, and duplication capabilities so content writers can curate libraries without digging into SQLite.
- When executions capture a prompt part, the metadata stores the part ID plus flattened instructions so downstream automations can reuse the formatting contract.

## CLI Utilities

The installed wheel exposes `prompt-manager`; the `python -m main` forms below remain equivalent repository-checkout commands. Bare `doctor` provides shallow readiness; `doctor catalog` audits records; `doctor config` and default `doctor embeddings` are focused offline views. Explicit `doctor embeddings --live` or `doctor analytics --live` may call a paid remote provider once. Default `doctor analytics` is a local counts report, not a health claim. Targeted prompt/chain routes use provider-free validators; standalone commands stay supported.

| Command | Purpose |
| --- | --- |
| `python -m main doctor [--json]` | Bounded read-only, provider-free readiness check for effective config, existing SQLite metadata, and unprobed embedding/model configuration. Missing first-run files and optional model settings warn; it neither repairs nor audits prompt records/Chroma. Exit 0 for completed OK/WARN, 1 for required FAIL, 2 for invalid usage, 3 for unexpected inspection error. Installed equivalent: `prompt-manager doctor [--json]`. |
| `python -m main doctor catalog [--json]` | Provider-free record audit of an existing SQLite prompt/chain catalog, without repository startup or mutations. Default text prints a summary plus sanitized issue lines (code, severity, generic meaning, record IDs); long per-issue ID lists show ten IDs and refer to `--json` for all. `--json` emits the complete versioned report. Neither mode includes prompt text or legacy issue messages. A nonempty WAL or unreadable schema fails closed; missing first-run DB warns. JSON can also precede `catalog`. Installed equivalent: `prompt-manager doctor catalog [--json]`. |
| `python -m main doctor index [--json]` | Provider-free, read-only exact-ID comparison of catalog prompts with saved embeddings and Chroma 1.5 SQLite metadata for the `prompt_manager` collection. Existing DB and metadata must be readable with no pending WAL/journal; missing first-run catalog/index warns, unsupported metadata fails closed. Reports counts only, never IDs or prompt text. Matching IDs are a best-effort observation, not a shared snapshot or proof of HNSW vector files, dimensions or search quality; repeat mismatches after closing writers. Never opens a Chroma client. |
| `python -m main doctor config [--details] [--json]` | Focused effective-settings health; details contain allowlisted source/availability flags only, no secret, DSN, or path values. Legacy `--print-settings` remains unchanged. |
| `python -m main doctor embeddings [--json] [--live]` | Offline by default: configuration readiness only. Explicit `--live` sends one synthetic text to the configured LiteLLM backend (network/cost possible), reports vector usability/dimension, and does not initialize or inspect the index. Legacy `diagnostics embeddings` retains its behavior. |
| `python -m main doctor analytics [--json] [--live] [--export-csv PATH]` | Default: provider-free, read-only aggregate of stored execution success/total counts (report, not health). Explicit `--live` additionally sends one synthetic request to the configured LiteLLM embedding backend after a readable report (network/cost possible; 15-second request timeout, no intentional retries); the report includes only probe status and dimension, not index health. Missing database warns; pending WAL/unreadable database fails closed without probing. Only explicit CSV export creates a new file; existing destination is never overwritten. The CSV may exist even if the subsequent probe fails. Legacy `diagnostics analytics` retains its own behavior. |
| `python -m main doctor prompt <uuid-or-exact-name> validate\|lint [--json]` | Provider-free, read-only targeted validation/lint via the immutable catalog loader. Exact names must resolve uniquely. Versioned output retains issue codes and counts, not prompt text or source-derived messages. |
| `python -m main doctor prompt <uuid-or-exact-name> test --suite PATH [--json]` | Run bounded local template fixtures with no provider calls or writes. Doctor permits only literal text and scalar `{{ variable }}` substitutions (not loops/filters/expressions/attributes); the separate legacy runner retains its prior template syntax. Reports counts and generic case failures without revealing body/fixture values. |
| `python -m main doctor chain <definition-file> validate [--json]` | Validate a local JSON chain definition using the production parser; reports step count or sanitized failure, without persistence or provider calls. |
| `python -m main catalog-export <path> [--format json\|yaml]` | Export prompts; YAML requires PyYAML (already bundled). |
| `python -m main catalog-import <path> [--dry-run] [--no-overwrite]` | Create or update prompts from a JSON catalogue file or directory. |
| `python -m main catalog-check [--json]` | Run a read-only, provider-free integrity pass over stored prompts and chains; reports duplicate names/bodies, invalid templates or references, missing stored embeddings, and broken chain prompt references. |
| `python -m main prompt-add [<path>\|--input-file path\|--json '{...}'\|--from-stdin\|inline fields] [--dry-run] [--no-overwrite]` | Add or update prompts through the catalog importer; use `--name`, `--description`, and `--prompt-text` for one inline prompt. Here, `--json` is an input payload, unlike output-format flags on read commands. |
| `python -m main prompt-show <prompt-id-or-name> [--json] [--full]` | Show one prompt resolved by UUID or one exact name. Default text is a readable operator view with a wrapped description and copy-ready `<prompt_body>` delimiters around the context; default JSON omits the full embedding vector and reports its presence/dimension, while `--json --full` emits the complete stored record including `ext4`. |
| `python -m main prompt-random` | Display one randomly selected local prompt in the standard readable prompt-detail view; it makes no provider calls or repository changes. |
| `python -m main prompt-find <query> [--limit N] [--category value] [--tag value] [--source value] [--active true\|false] [--json] [--full]` | Find prompts by the raw semantic ranking for the supplied natural-language query; explicit filters apply after ranking. Default `--json` omits raw `ext4` and reports embedding presence/dimension; `--json --full` emits complete stored records including embedding vectors. Unlike `suggest`, this command does not add intent hints or user-profile personalization. |
| `python -m main tag-list [--json]` | List each distinct logical prompt tag with total and active-prompt counts. Aggregation is local, deterministic, case-insensitive, and read-only. |
| `python -m main tag-show <tag> [--json]` | Show compact records for prompts with one exact logical tag, case-insensitively. A missing tag is a successful empty read. |
| `python -m main prompt-tag <prompt-id-or-name> add\|remove <tag> [--dry-run] [--json]` | Add or remove one non-blank prompt tag through the normal prompt lifecycle. Membership is case-insensitive and idempotent; `--dry-run` previews without writing. |
| `python -m main prompt-history <prompt-id-or-name> [--limit N] [--status success\|failed] [--window-days N] [--json] [--full]` | Inspect bounded read-only execution evidence for one uniquely resolved prompt. Default `--json` emits a compact prompt record without raw `ext4`; `--json --full` emits the complete stored prompt record including its embedding vector. Execution records are unchanged. |
| `python -m main suggest "search query"` | Run semantic retrieval and print top matches with intent metadata. |
| `python -m main usage-report [--path <file>]` | Summarize anonymized GUI analytics (counts, intents, recommendations). |
| `python -m main history-analytics [--window-days N --limit M --trend-window K]` | Display execution success rates, durations, ratings, and window/overall token totals for recent prompts. |
| `python -m main reembed` | Reset the ChromaDB collection and rebuild embeddings. Back up SQLite and Chroma first; rebuild may partially fail, and remote embedding providers may receive prompt text and incur cost. |
| `python -m main benchmark --prompt <uuid> [--model <id>] --request "…"` | Execute one or more prompts across configured LiteLLM models and compare duration/token usage alongside history stats. |
| `python -m main refresh-scenarios <uuid> [--max-scenarios N]` | Regenerate and persist scenario lists for a prompt via LiteLLM or the heuristic fallback. |
| `python -m main diagnostics <embeddings\|analytics> [options]` | Run embedding health checks or aggregated analytics diagnostics; `diagnostics <target> --help` lists target options. |
| `python -m main prompt-lineage <prompt-id-or-name> [--json]` | Inspect the read-only parent and child fork relationships for one uniquely resolved prompt; use UUID when names collide. |
| `python -m main prompt-fork <prompt-id-or-name> --name "..." [--commit-message "..."] [--json]` | Create a named prompt variant with preserved parent→child lineage while leaving the source asset unchanged; use UUID when names collide. |
| `python -m main prompt-restore-version <version-id> --confirm [--commit-message "..."] [--json]` | Restore one snapshot to the live prompt and record that result as a new version; `--confirm` is required and historical snapshots remain intact. |
| `python -m main prompt-version-diff <base-version-id> <target-version-id> [--json]` | Compare two snapshots of the same prompt, showing changed fields and a unified prompt-body diff without changing the live asset. |
| `python -m main prompt-version-list <prompt-id-or-name> [--limit N] [--json]` | List read-only version snapshots for one uniquely resolved prompt, including version number, snapshot ID, timestamp, parent version, and commit message; use UUID when names collide. |
| `python -m main prompt-compare <left-prompt-id-or-name> <right-prompt-id-or-name> [--json]` | Compare two current prompt assets without rendering or calling a model: state, selected metadata, Jinja variables, direct fork relation, persisted counters, and unified body diff. |
| `python -m main prompt-render <prompt-id-or-name> [--variables-json '{...}'\|--variables-file vars.json] [--validate-only] [--json]` | Render and validate local Jinja prompt variables without calling a model; `--validate-only` checks readiness without printing rendered text, and UUID is required when names collide. |
| `python -m main prompt-validate <prompt-id-or-name> [--json]` | Run a read-only, provider-free technical check for one prompt: required text, empty body, Jinja syntax, local related-prompt references, and blank/duplicate tags. It reports detected template variables but does not render or call a model. |
| `python -m main prompt-lint <prompt-id-or-name> [--json]` | Run read-only, provider-free advisory checks for one prompt: short description, missing action cue, unstructured long body, repeated instruction lines, and undocumented template-input context. Warnings are non-blocking; this does not validate syntax, render, call a model, or mutate data. |
| `python -m main prompt-test <prompt-id-or-name> --suite tests.json [--json]` | Run deterministic local Jinja-template fixtures against one prompt. The explicit JSON suite supplies unique case IDs, variable maps, and exact expected output; results report only case status/reasons and make no model or provider call. |
| `python -m main prompt-template-list [--json]` | Show all six effective built-in LiteLLM workflow templates in canonical order. Text mode uses bordered, numbered, wrapped blocks; `--json` emits labels, provenance (`default`/`override`), metrics, and complete text. It reads settings only and does not initialize services or call a model. |
| `python -m main prompt-chain-list` | List stored prompt chains (add `--include-inactive` for archived definitions). |
| `python -m main prompt-chain-show <uuid>` | Display a prompt chain with ordered steps and target prompts. |
| `python -m main prompt-chain-history [--chain-id <uuid>] [--limit N] [--json]` | Inspect bounded recent backend-managed prompt-chain run history. |
| `python -m main prompt-chain-export <uuid> <path>` | Export one stored prompt chain to a JSON file. |
| `python -m main prompt-chain-validate <path> [--json]` | Validate a prompt-chain JSON definition without persistence. |
| `python -m main prompt-chain-apply path/to/chain.json` | Create or update a prompt chain from a JSON definition (`name`, `description`, and ordered `steps` that only specify `prompt_id`, `order_index`, and optional `stop_on_failure`). |
| `python -m main prompt-chain-run <uuid> [--input "text"\|--input-file path] [--no-web-search]` | Execute a chain sequentially by feeding the provided plain-text input into the first step, automatically piping each response into the next step while optionally enriching every hop with live web context. |

| `python -m main --help` | Compact root help card grouped by task area; use `python -m main <command> --help` for authoritative command-specific options. |

### GUI Prompt Chain Manager

- Open the **Chain** tab (next to Template) in the main window to access the embedded manager, which reuses the full JSON import, CRUD, and run controls without a separate dialog.
- Use **New/Edit/Delete** in that tab to manage chains without touching JSON; the editor exposes name/description toggles, an ordered step table with prompt IDs, and a per-step “stop on failure” option—no templates or custom variables are required anymore.
- Prompt selection uses a searchable combo box populated from the prompt catalog, so you can reference prompts by name instead of pasting UUIDs.
- The left pane lists every stored chain (active + inactive) with refresh and JSON import controls; imports reuse the same validation as the CLI helper.
- The right pane surfaces description, ordered steps, and a plain-text field where you type the input that feeds the first step; that text is persisted per chain so you can rerun workflows quickly.
- Running a chain triggers the busy indicator (required for LLM work) while toast notifications confirm non-LLM actions such as refresh/import; results capture outputs plus per-step summaries for quick inspection.
- The Run Chain panel includes a persistent “Use web search” checkbox (default on) that mirrors the workspace toggle so each step can preload live context when a provider is configured; uncheck it to keep runs offline.

### Prompt Chain Definitions

`prompt-chain-apply` expects a JSON object that defines the chain plus ordered steps:

```json
{
  "id": "optional-uuid",
  "name": "Content Review Workflow",
  "description": "Extract, summarize, and critique content.",
  "steps": [
    {
      "order_index": 1,
      "prompt_id": "uuid-of-extract-prompt"
    },
    {
      "order_index": 2,
      "prompt_id": "uuid-of-review-prompt",
      "stop_on_failure": false
    }
  ]
}
```

Each step automatically receives the previous step’s response as its input, so you only provide the initial plain-text request when running a chain. Use `stop_on_failure: false` if a given prompt should not abort the flow on errors.

These commands share the same validation logic as the GUI; pass explicit paths as needed.

## Testing & Type Checking

- **Unit/Integration Tests**: `pytest -n auto --cov=core --cov-report=term-missing --cov-fail-under=80`.
- **Property-Based Tests**: Use `hypothesis` for prompt parsing, token-length sensitive logic, and JSON import/export features.
- **Type Checking**: The GitHub workflow runs `pyright main.py config models` with zero errors/warnings/informations. Full configured `pyright` is a separate local check; its 0.23.1 release-closeout result is historical, not an outstanding prerequisite. Do not infer branch-protection requirements from the workflow YAML alone.
- **Static Analysis**: `ruff check --fix .` (lint + autofix) followed by `ruff format --check .`.
- **Recommended workflow**:
  - before merge / push confidence check: run the individual workflow checks above; optionally run `nox -s format lint typecheck test` after installing Nox separately
  - during bounded feature work: targeted pytest + narrow Pyright on touched files first, then broaden only when the slice crosses shared seams
  - for roadmap/debt review: use `pyright`, `pyright gui --stats`, and `pyright tests --stats` as separately scoped inventory tools when they are not part of the chosen slice gate

## Maintenance, Telemetry & Analytics

- **Maintenance dialog**: Provides buttons to clear SQLite prompts, wipe ChromaDB embeddings, or reset all application data (usage logs, cache) with confirmation prompts and logging.
- **Snapshot backups**: **Create Backup Snapshot** in Maintenance zips the main SQLite file, Chroma directory, and a manifest to a user-selected path. It does not use SQLite's online-backup API, checkpoint/copy the SQLite WAL, or take an atomic cross-store snapshot. With active writers it may omit committed WAL data or capture inconsistent state. Do not rely on this GUI archive alone as a verified pre-`reembed` backup: stop all writers, obtain a consistent SQLite + Chroma copy, and verify restore before any destructive operation. An online-safe snapshot implementation would be a separate code change.
- **Category health panel**: Review per-category prompt counts, active prompt ratios, recent execution timestamps, and success rates directly inside the maintenance dialog; use the Refresh button after batch edits.
- **Telemetry**: ChromaDB anonymized telemetry is disabled in `core/prompt_manager/backends.py` (`anonymized_telemetry=False`). There is no supported `PROMPT_MANAGER_CHROMA_TELEMETRY` environment toggle in this implementation; changing telemetry behavior requires a separately scoped code/policy decision.
- **Usage analytics**: GUI intent workspace interactions are logged to `data/logs/intent_usage.jsonl` (timestamp, hashed query metadata, detected intents, top prompts). Disable via `gui.usage_logger.IntentUsageLogger` instantiation or by clearing the log path.
- **Analytics dashboard**: The GUI **Analytics** tab (and `python -m main diagnostics analytics`) pulls execution history, benchmark metadata, embeddings health, and intent usage logs into configurable charts. Set the window/prompt limits via the panel controls or CLI flags (`--window-days`, `--prompt-limit`), choose datasets (`usage`, `model_costs`, `benchmark`, `intent`, `embedding`), and export any dataset with `--export-csv` or the tab's **Export CSV** button for downstream BI tooling. A dedicated token summary above the dashboard mirrors the current window totals plus overall history so teams can reconcile spend while pivoting between datasets.

## Product Boundary

Before proposing major new features or expanding roadmap scope, read [`docs/product-ssot.md`](product-ssot.md).

Short version:
- PromptManager is a **local-first system for capturing, organizing, retrieving, inspecting, reusing, and refining prompt assets**.
- The core loop is **capture → normalize → retrieve → inspect → reuse → refine → optional trustworthy run support**.
- Execution, analytics, chains, and other advanced surfaces are supporting features only when they materially improve that loop.

Use the product SSOT to decide whether a change belongs in:
- core,
- supporting scope,
- later/frozen scope.

For active near-term priorities, use [`docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`](plans/2026-05-10-product-direction-ssot-next-cycle.md).
For delivered slice/history tracking, use [`docs/STATUS.md`](STATUS.md).
For the first bounded comparison between implementation and the older boundary framing, see [`docs/product-boundary-alignment-audit-2026-04-04.md`](product-boundary-alignment-audit-2026-04-04.md).

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
