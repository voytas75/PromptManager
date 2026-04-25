# PromptManager Roadmap Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Przełożyć aktualny product SSOT PromptManager na sekwencję małych wdrożeń, które po każdym kroku aktualizują plan, docs i SSOT tylko wtedy, gdy stan produktu realnie się zmienił.

**Architecture:** Plan jest kanonicznym dokumentem wykonawczym dla najbliższych prac roadmapowych. Każdy slice ma kończyć się trzema kontrolami: (1) kod i testy, (2) aktualizacja tego planu, (3) aktualizacja README / SSOT tylko wtedy, gdy zmienia się rzeczywisty stan produktu, język produktu albo priorytet roadmapy. Najpierw domykamy trust infrastructure, potem asset loop, potem structured runs, potem automation surfaces.

**Tech Stack:** Python 3.13, PySide6, pytest, PromptManager runtime settings stack, product docs under `docs/`.

---

## Canonical workflow rule

Po **każdej** implementacji wykonaj w tej kolejności:

1. zaktualizuj status odpowiedniego taska w tym pliku,
2. dopisz krótki `Implemented:` / `Verified:` note pod taskiem,
3. zaktualizuj `README.md` jeśli zmienił się user-visible product story,
4. zaktualizuj `docs/product-ssot.md` tylko jeśli zmienił się produktowy stan, priorytet albo definicja warstwy,
5. zaktualizuj `docs/product-roadmap-ssot.md` tylko jeśli zmieniła się kolejność lub definicja etapów,
6. uruchom adekwatne testy i zanotuj wynik w planie.

**Rule:** nie przepisywać SSOT dla kosmetycznych zmian implementacyjnych. SSOT aktualizować tylko przy zmianie produktu, nie przy samym postępie kodu.

---

## Current roadmap status

### Stage 1 — Trust and operability foundation
- [x] compact diagnostics with `OK` / `WARN` / `FAIL`
- [x] GUI/CLI diagnostics wording alignment — first slice
- [x] effective config source-of-value visibility
- [x] canonical precedence visibility in diagnostics output
- [x] fail-fast / blocking validation for critical runtime state
- [x] explicit routing/embedding provenance in user-visible summaries

### Stage 2 — Core prompt asset loop quality
- [x] inspect view as decision-support surface
- [x] stronger draft → asset promotion with duplicate/similarity judgment
- [x] search/recent/retrieval ergonomics improvements

### Stage 3 — Trustworthy prompt operations
- [x] structured runs v1 linked to prompt assets
- [x] baseline vs candidate comparison slice

### Stage 4 — Controlled automation surfaces
- [x] CLI/API parity on inspection and status surfaces
- [x] CLI/shared parity for run evidence and comparison cues
- [x] user-facing label parity across runtime/live preview surfaces
- [x] bounded operator recommendation cue from existing evidence

---

## Recommended execution order

1. effective config source-of-value
2. precedence visibility in diagnostics
3. startup fail-fast for blocking misconfig
4. inspect decision-support improvements
5. draft promotion duplicate/similarity support
6. retrieval ergonomics
7. structured runs v1
8. CLI/API parity expansion

---

### Task 1: Effective config source-of-value in diagnostics

**Status:** implemented

**Objective:** Pokazać w GUI/CLI nie tylko wartość konfiguracji, ale też jej źródło dla kluczowych runtime settings.

**Files:**
- Modify: `gui/runtime_settings_service.py`
- Modify: `cli/settings_summary.py`
- Modify: `gui/settings_dialog.py`
- Modify: `tests/test_runtime_settings_service.py`
- Modify: `tests/test_settings_dialog_diagnostics.py`
- Modify: `tests/test_settings_summary.py`
- Maintain: `docs/plans/2026-04-25-roadmap-implementation-plan.md`
- Update later if needed: `README.md`, `docs/product-ssot.md`, `docs/product-roadmap-ssot.md`

**Implementation target:**
Add a first user-visible `source` field for the most important diagnostics rows, starting with:
- fast model
- inference model
- embeddings
- API key presence

At minimum the source labels should support:
- `default`
- `config`
- `env`
- `runtime`
- `derived`
- `unknown` (only as safe fallback)

**Step 1: Write failing tests for diagnostics items with source metadata**

Add tests that expect diagnostics items to include a `source` key and user-facing detail/source consistency.

**Step 2: Run targeted tests to verify failure**

```bash
.venv/bin/pytest tests/test_gui_diagnostics_status.py tests/test_runtime_settings_service.py tests/test_settings_summary.py -q
```

**Step 3: Implement minimal source-aware diagnostics model**

Introduce a small internal helper / shape in `gui/runtime_settings_service.py` that can build items like:

```python
{
    "label": "Fast model",
    "status": "OK",
    "detail": "azure/gpt-4.1",
    "source": "env",
}
```

Do not over-generalize yet. Start with a bounded mapping for currently visible settings.

**Step 4: Surface source in CLI and GUI summaries**

Examples:
- CLI: `OK | Fast model: azure/gpt-4.1 [env]`
- GUI: `OK | Fast model: azure/gpt-4.1 (env)`

Prefer compact rendering.

**Step 5: Run targeted verification**

```bash
.venv/bin/pytest tests/test_gui_diagnostics_status.py tests/test_runtime_settings_service.py tests/test_settings_dialog_diagnostics.py tests/test_settings_summary.py tests/test_main_entry.py -q
```

**Step 6: Update this plan after implementation**

When green:
- change Task 1 status to `implemented`
- add `Implemented:` bullet with what was shipped
- add `Verified:` bullet with exact test command/result
- decide whether README or SSOT wording needs change

**Implemented:**
- Added `source` metadata to diagnostics items for GUI/CLI summaries.
- Surfaced compact source labels in CLI (`[config]`, `[default]`, etc.) and GUI (`(config)`, `(default)`, etc.).
- Wired first bounded source mapping for fast model, inference model, API key, embeddings, TTS, and Redis.

**Verified:**
- `.venv/bin/pytest tests/test_gui_diagnostics_status.py tests/test_runtime_settings_service.py tests/test_settings_dialog_diagnostics.py tests/test_settings_summary.py tests/test_main_entry.py -q`
- result: `37 passed`
- Extended compact user-visible provenance beyond diagnostics into runtime summaries, settings toasts, and live routing preview for routing + embeddings.
- `.venv/bin/pytest tests/test_runtime_settings_service.py tests/test_settings_dialog_live_preview.py tests/test_settings_workflow_summary_toast.py tests/test_settings_summary.py -q`
- result: `15 passed`

---

### Task 2: Canonical precedence visibility in diagnostics output

**Status:** implemented

**Objective:** Make the precedence model visible enough that a user can tell *why* a value won.

**Files:**
- Modified: `gui/runtime_settings_service.py`
- Modified: `cli/settings_summary.py`
- Modified: `tests/test_runtime_settings_service.py`
- Modified: `tests/test_settings_summary.py`
- Modified: `tests/test_gui_diagnostics_status.py`
- Modified: `tests/test_main_entry.py`
- Maintain: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Implementation target:**
Extend Task 1 from simple source labels to explicit precedence explanation for overridden values where useful.

**Done looks like:**
- diagnostics can explain when env overrides config
- derived/default values are clearly distinguished
- no silent ambiguity for core model-routing values

**Implemented:**
- Added bounded `precedence` text to diagnostics items for core runtime rows.
- Diagnostics now explain `env overrides config` when override context is available, otherwise show `env override in effect`.
- Distinguished `derived from fast model` vs `default value in effect` in shared diagnostics and CLI output.
- Extended CLI diagnostics rendering to print `[source; precedence]` when precedence context exists.

**Verified:**
- `.venv/bin/pytest tests/test_runtime_settings_service.py tests/test_settings_summary.py -q`
- result: `12 passed`
- `.venv/bin/pytest tests/test_gui_diagnostics_status.py tests/test_runtime_settings_service.py tests/test_settings_dialog_diagnostics.py tests/test_settings_summary.py tests/test_main_entry.py -q`
- result: `41 passed`

---

### Task 3: Fail-fast / blocking startup validation

**Status:** implemented

**Objective:** Distinguish blockers from warnings so invalid critical runtime state is not presented as merely advisory.

**Files:**
- Modified: `main.py`
- Modified: `tests/test_main_entry.py`
- Maintain: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Implementation target:**
Introduce bounded blocking-state detection for critical paths such as missing fast model / missing auth when runtime execution is expected.

**Implemented:**
- Added bounded startup blocker detection in `main.py` using the shared diagnostics contract.
- Normal startup now fails fast when core runtime execution prerequisites are missing: LiteLLM fast model or LiteLLM API key.
- Blocking validation applies only to default app startup; `--print-settings` and explicit CLI commands remain available for inspection and remediation workflows.
- Startup now prints a compact `Blocking configuration issues:` section plus a pointer to `--print-settings`.

**Verified:**
- `.venv/bin/pytest tests/test_main_entry.py::test_main_blocks_default_startup_when_critical_runtime_state_is_invalid tests/test_main_entry.py::test_main_allows_print_settings_even_when_runtime_state_is_blocking -q`
- result: `2 passed`
- `.venv/bin/pytest tests/test_gui_diagnostics_status.py tests/test_runtime_settings_service.py tests/test_settings_dialog_diagnostics.py tests/test_settings_summary.py tests/test_main_entry.py -q`
- result: `43 passed`

---

### Task 4: Inspect view as reuse/refine/fork decision surface

**Status:** implemented

**Objective:** Make inspect/detail view more useful for prompt decisions instead of raw metadata browsing.

**Implemented:**
- Added one bounded `Decision` cue to the shared detail widget so inspect now gives a direct next-step recommendation instead of only passive metadata.
- Wired `WorkspaceHistoryController` to derive decision-support wording from existing lineage state: `Fork before editing`, `Refine before reuse`, or `Reuse as-is`.
- Kept the slice bounded to existing seams and data sources: no new workflow, no new persistence, no shadow product model.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- result: `26 passed`
- `.venv/bin/pytest tests/test_canonical_operator_path_parity.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- result: `27 passed`

---

### Task 5: Draft → asset promotion with duplicate/similarity judgment

**Status:** implemented

**Objective:** Reduce prompt-library clutter by showing stronger duplicate and related-prompt cues during promotion.

**Implemented:**
- Strengthened the promote-draft flow so opening an existing strong match now carries explicit duplicate-judgment wording instead of a generic similar-match status.
- Added bounded status semantics in `PromptEditorFlow`: strong matches (`similarity >= 0.85`) now report `Opened likely duplicate instead of promoting.` while weaker matches keep the existing generic fallback.
- Kept the slice advisory-only and bounded to existing promote-time similarity data: no schema changes, no blocking behavior, no new workflow surface.

**Verified:**
- `.venv/bin/pytest tests/test_prompt_editor_flow.py::test_promote_draft_passes_similar_matches_and_can_open_existing -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_editor_flow.py tests/test_draft_promote_dialog.py -q`
- result: `16 passed`

---

### Task 6: Retrieval ergonomics improvements

**Status:** implemented

**Objective:** Improve recent/search/retrieval speed and legibility.

**Implemented:**
- Added focused retrieval-state coverage for `PromptListCoordinator.fetch_prompts(...)` so active search with zero matches stays distinct from the default full-catalog state.
- Added explicit search-error coverage so retrieval failures remain distinguishable from no-match searches instead of collapsing into the same empty-result shape.
- Kept the slice bounded to existing retrieval-state seams (`search_results`, `preserve_search_order`, `search_error`) without changing ranking, search backend behavior, or recent/search workflow surfaces.

**Verified:**
- `.venv/bin/pytest tests/test_prompt_list_coordinator.py -q`
- result: `4 passed`
- `.venv/bin/pytest tests/test_prompt_list_coordinator.py tests/test_recent_prompts.py tests/test_canonical_operator_path_parity.py -q`
- result: `6 passed`

---

### Task 7: Structured runs v1

**Status:** implemented

**Objective:** Record prompt-linked runs with enough provenance to support refinement decisions.

**Implemented:**
- Added persisted `run` provenance in execution context metadata for prompt executions: run kind, prompt id, prompt version, and conversation message count.
- Surfaced a bounded `Last run` evidence cue in inspect/detail flow using existing execution history metadata instead of a new UI-side model.
- Extended inspect/detail run evidence with a bounded `Candidate vs baseline` cue when two recent compatible runs expose rating + duration, so users can compare the latest candidate against the immediate baseline without opening a separate analytics flow.
- Kept the slice compact and inspect-oriented: latest run summary now shows status, model, prompt version, conversation message count, and duration when available.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q`
- result: `6 passed`
- `.venv/bin/pytest tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q`
- result: `12 passed`

### Task 8: CLI/API parity expansion

**Status:** implemented

**Objective:** Extend the same product model to headless/automation surfaces without creating a shadow model.

**Implemented:**
- Added bounded CLI analytics coverage that asserts the same shared analytics snapshot is rendered into headless diagnostics output, instead of introducing any parallel CLI-only model.
- Extended `test_main_entry.py` to verify execution summary plus model-cost and benchmark sections from the canonical snapshot fixture.
- Kept the slice intentionally narrow: no new API contract, no shadow serialization layer, no product-model fork.

**Verified:**
- `.venv/bin/pytest tests/test_main_entry.py::test_main_analytics_diagnostics_runs -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_main_entry.py tests/test_prompt_manager_execution.py -q`
- result: `36 passed`

## Update log

### 2026-04-25
- Created this canonical roadmap implementation plan.
- Confirmed current order remains: trust infrastructure → asset loop → structured runs → automation.
- Implemented Task 1: source-of-value labels in diagnostics for GUI/CLI.
- Verified Task 1 with targeted diagnostics/settings test suite: `37 passed`.
- Implemented Task 2: precedence explanations in shared diagnostics and CLI output.
- Verified Task 2 with targeted + related suite runs: `12 passed`, then `41 passed`.
- Implemented Task 3: bounded fail-fast startup validation for missing fast model / API key.
- Verified Task 3 with targeted startup tests and related trust suite: `2 passed`, then `43 passed`.
- Implemented Task 7: structured run provenance in execution context metadata.
- Verified Task 7 with targeted execution-history tests plus related tracker coverage: `10 passed`; pyright on touched files: `0 errors`.
- Extended Task 7: surfaced bounded `Last run` evidence in inspect/detail from persisted execution history provenance.
- Verified Task 7 evidence surface with targeted + nearby suites: `37 passed`, then `39 passed`.
- Extended Task 7 again: added bounded `Candidate vs baseline` comparison cue in inspect/detail summaries when two compatible runs exist.
- Verified bounded comparison cue with `tests/test_workspace_history_controller.py -q` (`6 passed`) and nearby UI smoke tests `tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q` (`12 passed`).
- Extended Stage 1 trust surfaces: added compact routing + embedding provenance in runtime summaries, settings toasts, and live routing preview.
- Verified routing/embedding provenance slice with `tests/test_runtime_settings_service.py tests/test_settings_dialog_live_preview.py tests/test_settings_workflow_summary_toast.py tests/test_settings_summary.py -q` (`15 passed`).
- Implemented Task 8: bounded CLI/headless parity coverage for shared analytics snapshot rendering.
- Verified Task 8 with targeted CLI analytics test: `1 passed`; related CLI/execution suite: `36 passed`.
