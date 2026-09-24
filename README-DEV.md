# README-DEV

Practical, short developer onboarding for PromptManager. For detailed engineering, CLI, maintenance, and quality-gate contracts, see [`docs/README-DEV.md`](docs/README-DEV.md); product direction and near-term priority remain owned by [`docs/product-ssot.md`](docs/product-ssot.md) and its [active plan](docs/plans/2026-05-10-product-direction-ssot-next-cycle.md).

This file exists to keep the main `README.md` focused on project introduction, what the product does, and how to get started.

## 1. What you are working on

PromptManager is a local-first desktop app for managing prompt assets.
As a developer, expect three main areas of work:
- GUI flows and dialogs in `gui/`
- CLI entrypoints and summaries in `cli/` plus `main.py`
- core product logic, settings, persistence, and retrieval in `core/`, `config/`, and `models/`

## 2. Repository map

Use this mental model when entering the repo:

- `main.py` — main application entrypoint
- `cli/` — CLI parsing, runtime helpers, summaries, non-GUI entry flows
- `gui/` — PySide6 dialogs, widgets, controllers, settings flows, main-window wiring
- `core/` — prompt manager logic, execution, retrieval, templating, maintenance, web search
- `config/` — settings model, defaults, config loading, persistence helpers
- `models/` — prompt/category/style data models
- `tests/` — pytest suite for CLI, GUI, settings, and product logic
- `docs/` — product SSOT, roadmap, rollout notes, and deeper documentation

## 3. Local setup

### Python

Project target:
- Python `3.13+`

### Option A — venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -e .[dev]
```

### Option B — uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e .[dev]
```

## 4. First-run checks

Start from the example environment file:

```bash
cp .env.example .env
```

PromptManager also supports JSON configuration via:
- `config/config.json`
- `PROMPT_MANAGER_CONFIG_JSON`

If you copy `config/config.template.json` to `config/config.json`, its LiteLLM embedding setting overrides the deterministic setting in the copied `.env.example`. For provider-free indexing/search, set `"embedding_backend": "deterministic"` in the JSON and confirm the effective value before use.

Important config note:
- if settings look wrong, verify whether `PROMPT_MANAGER_CONFIG_JSON` is overriding `config/config.json`
- if the path is explicitly exported to a missing file, settings fail closed; copying `.env.example` alone does not make its suggested path an explicit exported override
- for overlapping values, JSON takes precedence over canonical environment variables; inspect the effective settings rather than assuming an env change overrides JSON

Inspect effective configuration:

```bash
python -m main --no-gui --print-settings
```

The CLI summary is trust-first: it starts with the `Diagnostics` block (`Overall status`, source/precedence labels, and `Next steps` when needed) before lower-level path and provider sections.
It summarizes validated configuration and reported path state; it does not test path writability or Redis, LiteLLM, or ChromaDB connectivity. `doctor` offers bounded provider-free local checks; explicit `--live` probes can contact a provider and incur cost.

Launch the GUI:

```bash
python -m main
```

## 5. Daily commands

### Run the app

```bash
python -m main
```

### Print effective settings without GUI

```bash
python -m main --no-gui --print-settings
```

### Run the full test suite

```bash
.venv/bin/pytest -q
```

### Run one focused test file

```bash
.venv/bin/pytest tests/test_settings_summary.py -q
```

### Run Ruff

```bash
.venv/bin/ruff check .
```

### Run Pyright

```bash
.venv/bin/pyright
```

This is the full configured strict local check. The GitHub Quality Gates workflow currently runs the narrower `.venv/bin/pyright main.py config models`; see [`docs/README-DEV.md`](docs/README-DEV.md) for the distinction. The workflow's existence alone does not prove GitHub branch protection is enabled.

## 6. Quality gates used by the repo

Defined in `pyproject.toml`:
- `pytest`
- `ruff`
- `pyright`
- strict type checking (`typeCheckingMode = "strict"`)
- coverage threshold (`fail_under = 80`)

At minimum before considering work done:
1. run the focused tests for your changed area
2. run broader tests if the change touches shared logic
3. run `ruff` if you changed Python code
4. run `pyright` if you changed types, settings, GUI wiring, or public interfaces

## 7. How to work by area

### If you change GUI
Typical files live in:
- `gui/`
- `gui/widgets/`
- `gui/dialogs/`
- `gui/controllers/`

Then usually run:
```bash
python -m main
.venv/bin/pytest tests/test_settings_dialog_diagnostics.py -q
```

### If you change CLI or startup behavior
Typical files live in:
- `main.py`
- `cli/`

Then usually run:
```bash
python -m main --no-gui --print-settings
.venv/bin/pytest tests/test_main_entry.py -q
```

### If you change settings or config resolution
Typical files live in:
- `config/settings.py`
- `config/`
- `gui/runtime_settings_service.py`
- `cli/settings_summary.py`

Then usually run:
```bash
python -m main --no-gui --print-settings
.venv/bin/pytest tests/test_runtime_settings_service.py tests/test_settings_summary.py -q
```

### If you change core prompt logic
Typical files live in:
- `core/`
- `models/`

Then run the nearest focused tests in `tests/` first, then widen scope if shared behavior changed.

## 8. Practical feature notes

PromptManager includes, among other things:
- prompt catalog CRUD, tagging, editing, and fork/version flows
- Quick Capture, Draft Promote, recent reopen, and inspection cues
- semantic retrieval via ChromaDB with configured embedding backends
- Jinja2 template preview with validation support
- optional LiteLLM-backed workspace and related CLI helpers
- lightweight history and diagnostics surfaces

Treat the main product center as prompt-asset management first.
If a change increases product breadth but weakens clarity, trust, or the core prompt loop, it is probably the wrong priority.

## 9. Product and documentation links

Developer-facing docs:
- `docs/product-ssot.md`
- `docs/canonical-usage-path-v1.md`

Use `docs/` for detailed rollout plans or implementation-specific notes that would make this file noisy.

## 10. Maintenance rule

If a detail is mainly useful to contributors rather than first-time GitHub visitors, put it here or under `docs/`, not in the main `README.md`.
