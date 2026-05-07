# PromptManager — Prompt Chain Implementation Plan

Status: supporting note
Owner: Wojtek / Prompt Manager Team
Updated: 2026-05-07 (superseded as active execution ledger by rollout plan)
Feature SSOT: `docs/plans/2026-05-06-prompt-chain-ssot.md`
Execution ledger: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`
Canonical product SSOT: `docs/product-ssot.md`

## Purpose

This file is a historical supporting ledger for earlier prompt-chain slices.

It no longer serves as the active execution ledger.
The active implementation truth now lives in `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`.
Keep this file only as a narrow record of already delivered slices so future sessions do not mistake it for the current next-step source.

---

## Planning rules

1. Keep prompt chains as a bounded supporting feature.
2. Do not widen into workflow-engine semantics.
3. Prefer one seam per slice.
4. Prefer operator trust, inspectability, and lifecycle completeness over extra engine power.
5. Update this plan immediately after each implemented slice.
6. Update feature SSOT only when feature truth changes, not just because code moved.

---

## Confirmed baseline before this plan

Treat the following as already delivered unless a focused regression proves otherwise:
- GUI chain CRUD exists.
- CLI chain list/show/apply/run/validate/export exists.
- `prompt-chain-show --json` exists.
- `prompt-chain-run --json` exists.
- Chains accept one plain-text input.
- Later steps receive previous step output.
- Optional web-search enrichment exists.
- Optional summary exists.
- Run results distinguish `step_outputs`, `final_output_text`, and `final_summary_text`.
- The bounded clarity/readability cycle in `docs/plans/2026-04-29-prompt-chain-clarity-continuity-roadmap.md` is complete.

This plan must not reopen those as fake new feature work.

---

## Execution order

This plan follows the current feature SSOT order after the completed lifecycle, metadata-visibility, and step-semantics slices:

1. result contract tightening
2. result-surface completeness
3. model clarity cleanup
4. editor ergonomics
5. bounded history

---

## Slice ledger

### Slice 1 — Prompt-chain model clarity v1

**Status:** completed

**Goal:**
Align the active prompt-chain data model with the real linear plain-text runtime so the codebase stops presenting historical fields as if they were active feature semantics.

**Primary seams:**
- `models/prompt_chain_model.py`
- `core/prompt_manager/chains.py`
- `gui/dialogs/prompt_chain_editor.py`
- `gui/dialogs/prompt_chains.py`
- `cli/commands.py`
- focused chain tests

**Expected work:**
- audit which prompt-chain fields are truly active vs legacy,
- reduce or isolate legacy fields from active product semantics,
- keep backward-compatible import handling only where necessary,
- make CLI/GUI labels reflect actual runtime meaning,
- avoid widening execution behavior.

**Boundaries:**
- no branching,
- no new workflow semantics,
- no history subsystem yet,
- no broad migration framework unless strictly required by existing persistence.

**Suggested verification:**
- focused `tests/test_prompt_chain_model.py`
- focused `tests/test_prompt_chain_dialog.py`
- focused `tests/test_prompt_chain_cli.py`
- `ruff check` on touched files
- `pyright` on touched files

**Implemented:**
- `PromptChainStep` now derives runtime-safe defaults for legacy-looking step fields (`input_template`, `output_variable`, `condition`) instead of requiring them as active constructor semantics.
- `chain_from_payload()` now treats `input_template`, `output_variable`, and `condition` as compatibility-only import fields; active runtime values are normalized back to the canonical linear model.
- Legacy import values are preserved only under step metadata as `legacy_runtime_fields` when present, so compatibility survives without keeping those fields in the active model narrative.
- Focused model tests now lock both behaviors.

**Verified:**
- `pytest tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `ruff check models/prompt_chain_model.py tests/test_prompt_chain_model.py`
- `pyright` *do weryfikacji in this environment* (`pyright: command not found`)

---

### Slice 2 — Prompt-chain CLI inspect/export lifecycle v1

**Status:** completed

**Goal:**
Complete the bounded operator lifecycle for inspecting and exporting prompt chains from CLI.

**Primary seams:**
- `cli/commands.py`
- `cli/parser.py`
- `models/prompt_chain_model.py`
- focused CLI tests

**Expected work:**
- add `prompt-chain-show --json`,
- add `prompt-chain-show --verbose`,
- add `prompt-chain-export`,
- keep output deterministic and operator-readable,
- keep exported payload aligned with canonical active chain semantics.

**Boundaries:**
- no API layer,
- no batch orchestration,
- no GUI work in this slice,
- no run-history work yet.

**Suggested verification:**
- focused `tests/test_prompt_chain_cli.py`
- add/export/show-specific CLI tests
- `ruff check cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `ruff format --check ...`
- `pyright` on touched files

**Implemented:**
- Added `prompt-chain-show --json` to the CLI parser.
- Added deterministic JSON output in `run_prompt_chain_show()` with bounded chain metadata and ordered step records.
- Kept the payload aligned with active prompt-chain semantics: chain identity/status, summarize flag, and per-step prompt/failure metadata without reviving legacy runtime fields.
- Added a focused CLI regression test covering machine-readable JSON inspection output.

**Verified:**
- `pytest tests/test_prompt_chain_cli.py -q`
- `pytest tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `ruff check cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `python -m py_compile cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `pyright` *do weryfikacji in this environment* (`pyright: command not found`)

---

### Slice 3 — Prompt-chain export v1

**Status:** completed

**Goal:**
Complete the bounded operator lifecycle by allowing prompt chains to be exported as deterministic JSON aligned with canonical active semantics.

**Primary seams:**
- `cli/commands.py`
- `cli/parser.py`
- `models/prompt_chain_model.py`
- focused CLI/model tests

**Expected work:**
- add `prompt-chain-export`,
- emit deterministic JSON payloads for one selected chain,
- keep exported fields aligned with canonical active chain semantics,
- avoid reviving legacy runtime fields as first-class product semantics.

**Boundaries:**
- no GUI work,
- no API layer,
- no batch export orchestration,
- no run-history work yet.

**Suggested verification:**
- focused `tests/test_prompt_chain_cli.py`
- export-specific CLI tests
- `ruff check cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `ruff format --check ...`
- `pyright` on touched files

**Implemented:**
- Added `prompt-chain-export <chain_id> <path>` to the CLI parser.
- Added `run_prompt_chain_export()` to write a deterministic single-chain JSON artifact.
- Kept exported payloads aligned with active prompt-chain semantics: chain identity, active/summarize flags, and ordered step prompt/failure metadata only.
- Registered the command in the CLI command map and added a focused export regression test.

**Verified:**
- `pytest tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `ruff check cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `python -m py_compile cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `pyright` *do weryfikacji in this environment* (`pyright: command not found`)

---

### Slice 4 — Prompt-chain definition validation v1

**Status:** completed

**Goal:**
Allow operators to validate chain definitions before persistence.

**Primary seams:**
- `cli/commands.py`
- `cli/parser.py`
- `models/prompt_chain_model.py`
- possibly shared validation helpers
- focused CLI/model tests

**Expected work:**
- add `prompt-chain-validate` or equivalent `prompt-chain-apply --dry-run`,
- validate payload shape and active semantic rules,
- surface explicit errors for malformed or misleading definitions,
- keep behavior bounded to definition inspection, not runtime changes.

**Boundaries:**
- no new persistence model,
- no GUI validation wizard,
- no schema-engine comeback.

**Suggested verification:**
- focused model + CLI tests
- `ruff check`
- `pyright` on touched files

**Implemented:**
- added `prompt-chain-validate <path>` as a manager-free CLI command,
- extracted shared `_validate_prompt_chain_payload()` helper reused by validate/apply,
- kept validation bounded to JSON object loading plus `chain_from_payload()` semantic checks,
- added focused CLI coverage for valid definitions and empty-step rejection.

**Verified:**
- `pytest tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `ruff check cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `python -m py_compile cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `pyright cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py` *do weryfikacji in this environment (`pyright: command not found`)*

---

### Slice 5 — Prompt-chain run result semantics v1

**Status:** completed

**Goal:**
Separate final raw output, final summary, and per-step outputs so GUI and CLI stop presenting one ambiguous terminal result concept.

**Primary seams:**
- `core/prompt_manager/chains.py`
- `gui/dialogs/prompt_chains.py`
- `cli/commands.py`
- focused chain backend/dialog/CLI tests

**Expected work:**
- add explicit final raw output field,
- keep final summary distinct,
- make GUI wording and CLI wording semantically honest,
- preserve bounded linear-run behavior.

**Boundaries:**
- no analytics expansion,
- no new engine behavior,
- no run-history storage in this slice.

**Suggested verification:**
- `tests/test_prompt_chain_backend.py`
- `tests/test_prompt_chain_dialog.py`
- `tests/test_prompt_chain_cli.py`
- `ruff check`
- `pyright` on touched files

**Implemented:**
- `PromptChainRunResult` now separates `step_outputs`, `final_output_text`, and `final_summary_text`
- backend derives `final_output_text` from the last successful step without changing execution flow
- GUI result view now labels sections as `Step outputs`, `Final output`, and `Final summary`
- CLI `prompt-chain-run` now prints distinct final output vs final summary sections
- focused tests updated to lock the new result semantics and wording

**Verified:**
- `pytest tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py tests/test_prompt_chain_backend.py -q`
- `37 passed in 1.06s`
- `ruff check` *do weryfikacji*
- `pyright` *do weryfikacji*

---

### Slice 6 — Prompt-chain run JSON output v1

**Status:** completed

**Goal:**
Expose machine-readable run results for prompt-chain execution while preserving readable console output for normal operators.

**Primary seams:**
- `cli/commands.py`
- `cli/parser.py`
- `core/prompt_manager/chains.py`
- focused CLI/backend tests

**Expected work:**
- add `prompt-chain-run --json`,
- include chain identity, chain input, final raw output, final summary, step statuses, and bounded metadata,
- keep JSON deterministic and stable.

**Boundaries:**
- no remote API contract,
- no persistence redesign,
- no broad report/export matrix yet.

**Suggested verification:**
- focused CLI/backend tests
- `ruff check`
- `pyright` on touched files

**Implemented:**
- added `prompt-chain-run --json` flag in `cli/parser.py`
- `run_prompt_chain_run()` now emits deterministic JSON when `--json` is set
- JSON payload includes chain identity, chain input, `final_output_text`, `final_summary_text`, `step_outputs`, and bounded per-step details
- normal human-readable CLI output remains unchanged when `--json` is not requested
- focused CLI tests now lock both JSON and text output modes

**Verified:**
- `pytest tests/test_prompt_chain_cli.py -q`
- `7 passed in 0.52s`
- `pytest tests/test_prompt_chain_cli.py tests/test_prompt_chain_backend.py tests/test_prompt_chain_dialog.py -q`
- `38 passed in 1.19s`
- `ruff check` *do weryfikacji*
- `pyright` *do weryfikacji*

---

### Slice 7 — Prompt-chain execution metadata visibility v1

**Status:** completed

**Goal:**
Make per-step execution evidence clearer without adding engine complexity.

**Primary seams:**
- `core/prompt_manager/chains.py`
- `gui/dialogs/prompt_chains.py`
- `cli/commands.py`
- focused backend/dialog/CLI tests

**Expected work:**
- expose richer step-run metadata,
- include prompt identity/name where justified,
- include duration/status,
- include whether web search was requested/applied,
- include bounded skip/failure reasons.

**Boundaries:**
- no chain history store yet,
- no dashboarding,
- no new orchestration semantics.

**Suggested verification:**
- focused backend/dialog/CLI tests
- `ruff check`
- `pyright` on touched files

**Implemented:**
- extended `PromptChainStepRun` with operator-facing execution metadata: `prompt_name`, `request_text`, `response_text`, `duration_ms`, and bounded web-search flags
- backend `run_prompt_chain()` now records per-step prompt identity, enriched request text, step response text, duration, and whether web search was requested/applied
- CLI `prompt-chain-run` JSON output now includes the richer per-step metadata, and text mode now prints readable step evidence instead of preview-only status lines
- GUI run result rendering now surfaces prompt name, duration, and web-search evidence inside each step block while preserving the bounded linear run presentation
- added focused backend, CLI, and dialog regression coverage for the new metadata surface

**Verified:**
- `pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `40 passed in 1.13s`
- `ruff check core/prompt_manager/chains.py cli/commands.py gui/dialogs/prompt_chains.py tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py`
- `python -m py_compile core/prompt_manager/chains.py cli/commands.py gui/dialogs/prompt_chains.py tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py`
- `pyright` *do weryfikacji in this environment*

---

### Slice 8 — Prompt-chain step-output semantics v1

**Status:** completed

**Goal:**
Make per-step outputs easier to interpret for operators without weakening deterministic machine-readable results.

**Primary seams:**
- `core/prompt_manager/chains.py`
- `gui/dialogs/prompt_chains.py`
- `cli/commands.py`
- focused backend/dialog/CLI tests

**Expected work:**
- expose clearer human-facing step labels,
- surface `output_variable` or equivalent step identity where it improves inspection,
- keep stable machine-readable `step_outputs`,
- avoid changing the linear run model.

**Boundaries:**
- no branching semantics,
- no payload model expansion beyond bounded inspection value,
- no history subsystem yet.

**Suggested verification:**
- focused backend/dialog/CLI tests
- `ruff check`
- `pyright` on touched files

**Implemented:**
- extended `PromptChainStepRun` with `step_label` and `step_output_key` so human-facing step identity is explicit instead of inferred ad hoc in each surface
- backend `run_prompt_chain()` now computes a stable machine key (`step_{n}`) plus a human label (`output_variable` when present, otherwise the machine key) for every step
- backend `step_outputs` now preserves the stable machine key while also surfacing `output_variable` aliases when present, keeping deterministic machine reads without hiding operator-friendly labels
- CLI `prompt-chain-run --json` now includes `step_label` and `step_output_key`, and text output now renders both values in step summaries
- GUI step rendering now shows the human label in the title and the machine output key as a separate detail line in both plaintext and rich views
- added focused backend, CLI, and dialog regression coverage for the new step-identity semantics

**Verified:**
- `pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `42 passed in 1.11s`
- `ruff check core/prompt_manager/chains.py cli/commands.py gui/dialogs/prompt_chains.py tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py`
- `python -m py_compile core/prompt_manager/chains.py cli/commands.py gui/dialogs/prompt_chains.py tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py`
- `pyright` *do weryfikacji in this environment* (`pyright: command not found`)

---

### Slice 9 — Prompt-chain result-surface completeness v1

**Status:** completed

**Goal:**
Improve bounded result consumption ergonomics now that result semantics and run JSON exist.

**Primary seams:**
- `cli/parser.py`
- `cli/commands.py`
- focused CLI tests

**Expected work:**
- add `prompt-chain-run --final-output-only`,
- add `prompt-chain-run --summary-only`,
- add `prompt-chain-run --status-only`,
- keep machine-readable and human-readable modes bounded and deterministic.

**Boundaries:**
- no reporting framework,
- no API matrix,
- no new engine semantics,
- no broad export redesign,
- no GUI changes in this slice.

**Suggested verification:**
- focused CLI tests
- focused backend/dialog regression sweep to confirm no result-surface regressions
- `ruff check`
- `pyright` on touched files

**Implemented:**
- added `prompt-chain-run --final-output-only` to emit only `final_output_text`
- added `prompt-chain-run --summary-only` to emit only `final_summary_text`
- added `prompt-chain-run --status-only` to emit only a bounded derived final chain status (`success`, `failed`, or `skipped`)
- kept existing `--json` payload and default human-readable output unchanged
- added focused CLI regression coverage for all three single-surface output modes

**Verified:**
- `pytest tests/test_prompt_chain_cli.py -q`
- `10 passed in 0.54s`
- `pytest tests/test_prompt_chain_cli.py tests/test_prompt_chain_backend.py tests/test_prompt_chain_dialog.py -q`
- `45 passed in 1.33s`
- `ruff check cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `python -m py_compile cli/commands.py cli/parser.py tests/test_prompt_chain_cli.py`
- `pyright` *do weryfikacji in this environment (`pyright: command not found`)*

---

### Slice 10 — Prompt-chain editor ergonomics v1

**Status:** completed

**Goal:**
Improve editing ergonomics without turning the GUI into a visual workflow builder.

**Primary seams:**
- `gui/dialogs/prompt_chain_editor.py`
- focused dialog tests

**Expected work:**
- duplicate chain,
- step reorder controls,
- clearer prompt preview while editing,
- bounded warnings/validation where they improve trust.

**Boundaries:**
- no graph canvas,
- no drag-and-drop workflow builder unless a very small bounded control is clearly justified,
- no execution-engine changes unless directly required by editor truthfulness.

**Suggested verification:**
- focused `tests/test_prompt_chain_dialog.py`
- `ruff check`
- `pyright` on touched files

**Implemented:**
- added bounded step reorder controls (`Move Up`, `Move Down`) to `PromptChainEditorDialog`
- selection-aware button state now disables invalid reorder actions for the first and last step
- editor preserves selected row across table refreshes so reorder actions remain stable and predictable
- added focused dialog regression coverage for moving steps up/down and for reorder button-state behavior
- existing save path continues to normalize final `order_index` values on save
- added a `Duplicate` action in `PromptChainManagerPanel` for copying the currently selected chain into the editor before saving
- duplicate flow creates a new chain id, new step ids, preserves prompt references and step order, and seeds the editor with the copied chain named `{original} (Copy)`
- added a read-only `Prompt preview` pane to `PromptChainStepDialog` so operators can inspect the selected prompt body while editing a chain step
- prompt preview now updates on prompt selection change, shows prompt name/category, truncates overly long bodies, and falls back cleanly when the prompt body is unavailable in the current catalog
- added a non-blocking warning label in `PromptChainEditorDialog` when the same prompt is reused across multiple steps, to make repeated execution intent explicit before save

**Verified:**
- `pytest tests/test_prompt_chain_dialog.py -q`
- `41 passed in 1.32s`
- `ruff check gui/dialogs/prompt_chain_editor.py tests/test_prompt_chain_dialog.py`
- `All checks passed!`
- `.venv/bin/pyright gui/dialogs/prompt_chain_editor.py tests/test_prompt_chain_dialog.py`
- `do weryfikacji jako osobny debt slice: strict test-file violations (głównie reportPrivateUsage w tests/test_prompt_chain_dialog.py), bez potwierdzonej regresji reorder-controls`

---

### Slice 11 — Prompt-chain bounded history v1

**Status:** completed

**Goal:**
Add lightweight chain-run history only if earlier slices prove the run evidence model is stable enough.

**Primary seams:**
- `core/prompt_manager/chains.py`
- repository/history seams to be identified during implementation
- `gui/dialogs/prompt_chains.py`
- `cli/commands.py`
- focused history tests

**Expected work:**
- store a lightweight chain-run record,
- expose bounded recent inspection,
- keep the scope tightly tied to trust/debug value.

**Boundaries:**
- no analytics-first dashboard,
- no scheduler system,
- no long-running automation platform,
- no chain-centric product repositioning.

**Suggested verification:**
- focused repository/backend/dialog/CLI tests
- `ruff check`
- `pyright` on touched files

**Implemented:**
- added session-scoped bounded recent chain-run history in `PromptChainManagerPanel`
- each successful GUI run now records a lightweight entry with chain name, bounded input preview, aggregate status, and local time cue
- surfaced the last 5 session runs in a read-only summary label under execution results
- kept scope intentionally local to GUI session memory only; no repository persistence, CLI history, or analytics additions
- fixed a brittle delete-dialog test assertion while adding history coverage

**Verified:**
- `pytest tests/test_prompt_chain_dialog.py -q`
- `38 passed in 1.09s`
- `pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `55 passed in 1.16s`
- `ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py`
- `python -m py_compile gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py`
- `pyright` *do weryfikacji in this environment (`pyright: command not found`)*

---

## Delivery workflow per slice

For each slice:
1. choose one slice only,
2. add or update the nearest focused tests first,
3. implement the smallest change that satisfies the slice,
4. run focused verification,
5. update this plan:
   - change slice status,
   - fill `Implemented:`,
   - fill `Verified:`,
6. update `docs/CHANGELOG.md` only if shipped behavior changed,
7. revise prompt-chain SSOT only if feature truth changed.

---

## Status vocabulary

Use only:
- `pending`
- `in_progress`
- `implemented`
- `completed`
- `covered by existing behavior`
- `cancelled`

Avoid vague labels.

---

## Current recommended next slice

**Slice 13 — Prompt-chain GUI result actions v1**

Why this is next:
- backend contract and CLI artifact save flow are already explicit,
- the next bounded usability gap is GUI-side copy/save actions for the surfaced final result,
- it improves result consumption without widening chain semantics.

---

## Definition of done for this plan

This implementation plan remains healthy only if:
- it reflects the real next slice,
- completed slices are marked immediately,
- stale pending items are not left behind after behavior ships,
- future prompt-chain work continues to respect the bounded feature SSOT.
