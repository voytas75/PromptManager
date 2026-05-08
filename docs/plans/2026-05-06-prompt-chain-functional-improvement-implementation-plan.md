# PromptManager Prompt Chain Functional Improvement Implementation Plan

Status: supporting note
Execution ledger: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`
Purpose: narrow shipped ledger for the completed result-contract, artifact-save, and GUI result-actions slices.


> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Tighten prompt-chain result semantics, clean up model drift, and improve result consumption/history without expanding PromptManager into a workflow engine.

**Architecture:** Keep the current bounded linear runner intact. Prioritize explicit output/result contracts, small CLI/GUI consumption improvements, and compatibility-safe model cleanup. Persist only lightweight evidence that improves trust/debug value.

**Tech Stack:** Python, PromptManager core/CLI/GUI layers, pytest, Ruff, existing prompt-chain SSOT docs.

---

## Scope guardrails

This plan is allowed to improve:
- result contract clarity,
- bounded result consumption,
- compatibility-safe model clarity,
- lightweight chain-run evidence,
- small GUI ergonomics around result handling.

This plan must not introduce:
- branching workflows,
- loops,
- retry workflow semantics,
- scheduler behavior,
- multi-agent orchestration,
- visual workflow builder UX,
- dashboard-first expansion.

---

## Confirmed current state

Treat these as already delivered unless focused regression proves otherwise:
- linear prompt-chain execution over one plain-text input,
- GUI chain CRUD and run inspection,
- CLI `list/show/apply/validate/export/run`,
- `prompt-chain-run --json`,
- `prompt-chain-run --final-output-only`,
- `prompt-chain-run --summary-only`,
- `prompt-chain-run --status-only`,
- per-step `step_label` and `step_output_key`,
- `final_output_text` vs `final_summary_text`,
- session-scoped recent GUI history for chain runs.

Do not re-implement these as if they were missing.

---

## Phase 1 — Result contract tightening

### Task 1: Document the canonical machine-vs-alias output contract

**Objective:** Make the output-key contract explicit before changing payload semantics.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-next-slices-plan.md`
- Reference: `core/prompt_manager/chains.py`
- Reference: `cli/commands.py`

**Steps:**
1. Update the SSOT to state explicitly that machine consumers should rely on one canonical output identity per step.
2. Define alias semantics as operator-facing convenience, not equal machine-contract authority.
3. Record the expected direction for JSON/run payloads:
   - canonical `step_output_key`,
   - explicit alias field or alias mapping,
   - explicit final-step identity.
4. Keep docs compatible with current code until the implementation slice lands; do not claim behavior that is not yet shipped.

**Verification:**
- Read the changed SSOT sections and confirm they distinguish:
  - canonical machine key,
  - operator alias,
  - final-step identity.

---

### Task 2: Add failing backend/CLI tests for canonical output contract v1

**Status:** completed

**Objective:** Freeze the desired contract in tests before refactoring payload generation.

**Files:**
- Modify: `tests/test_prompt_chain_backend.py`
- Modify: `tests/test_prompt_chain_cli.py`
- Reference: `core/prompt_manager/chains.py`
- Reference: `cli/commands.py`

**Step 1: Write failing backend tests**
Add focused tests for:
- `step_outputs` exposing canonical machine keys only,
- alias metadata exposed separately,
- final-step identity fields being derivable and stable.

Suggested assertions shape:
- `result.step_outputs == {"step_1": "...", "step_2": "..."}`
- alias metadata maps `output_variable -> step_output_key`
- final-step contract identifies the last successful step.

**Step 2: Write failing CLI JSON tests**
Add tests asserting JSON includes:
- `final_step_output_key`,
- optional `final_step_label`,
- explicit alias field or alias mapping,
- no implicit requirement to consume alias keys from `step_outputs`.

**Step 3: Run focused tests to confirm RED**
Run:
- `pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py -q`
Expected: FAIL on the new contract assertions.

**Step 4: Commit after GREEN later**
Use later when task is complete.

**Implemented:**
- added focused backend regression coverage locking `step_outputs` to canonical machine keys only,
- added backend assertions for explicit alias metadata and final-step identity fields,
- updated CLI run JSON tests to require `step_aliases`, `final_step_id`, `final_step_output_key`, and `final_step_label`,
- confirmed RED before implementation with the focused backend/CLI pack.

**Verified:**
- `pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py -q`
- RED confirmed before GREEN on the new contract assertions.

---

### Task 3: Implement canonical output contract in core result objects

**Status:** completed

**Objective:** Make backend result semantics deterministic for automation without breaking the bounded runner model.

**Files:**
- Modify: `core/prompt_manager/chains.py`
- Modify: `core/__init__.py` only if exported symbols need adjustment
- Modify: `tests/test_prompt_chain_backend.py`

**Implementation targets:**
- keep `step_output_key = step_{n}` as canonical machine identity,
- stop using alias keys as equal peers inside `step_outputs`,
- add explicit alias metadata, for example one of:
  - per-step alias field only, or
  - a run-level `step_aliases` mapping,
- add explicit final-step contract fields, for example:
  - `final_step_id`,
  - `final_step_output_key`,
  - `final_step_label`.

**Constraint:**
Choose the smallest contract extension that keeps automation deterministic.
Do not redesign the whole execution object graph.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_backend.py -q`
- `python -m py_compile core/prompt_manager/chains.py tests/test_prompt_chain_backend.py`

Expected:
- tests pass,
- no syntax errors.

**Implemented:**
- extended `PromptChainRunResult` with explicit alias and final-step identity fields,
- `run_prompt_chain()` now keeps `step_outputs` canonical-machine-key only,
- alias semantics are exposed separately via run-level `step_aliases`,
- final terminal step identity is now surfaced as `final_step_id`, `final_step_output_key`, and `final_step_label`.

**Verified:**
- `pytest tests/test_prompt_chain_backend.py -q`
- `python -m py_compile core/prompt_manager/chains.py tests/test_prompt_chain_backend.py`

---

### Task 4: Surface the tightened contract in CLI JSON output

**Status:** completed

**Objective:** Make CLI JSON reflect the canonical backend contract clearly.

**Files:**
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Implementation targets:**
- include explicit final-step identity fields in JSON,
- include alias semantics explicitly,
- keep `step_outputs` deterministic and machine-readable,
- preserve human-readable CLI text output.

**Constraint:**
Do not add a second competing JSON shape.
Keep one primary structured contract.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
- `ruff check cli/commands.py tests/test_prompt_chain_cli.py`

Expected:
- tests pass,
- Ruff passes.

**Implemented:**
- CLI `prompt-chain-run --json` now exposes `step_aliases` explicitly,
- JSON now includes `final_step_id`, `final_step_output_key`, and `final_step_label`,
- `step_outputs` remains the single canonical machine-readable output mapping,
- preserved the existing human-readable text output path.

**Verified:**
- `pytest tests/test_prompt_chain_cli.py -q`
- `ruff check cli/commands.py tests/test_prompt_chain_cli.py`

---

## Phase 2 — Result artifact handling

### Task 5: Add failing CLI tests for saving run results to a file

**Status:** completed

**Objective:** Lock the desired save-to-file behavior before CLI changes.

**Files:**
- Modify: `tests/test_prompt_chain_cli.py`
- Reference: `cli/parser.py`
- Reference: `cli/commands.py`

**Step 1: Add tests for `--output-file`**
Cover:
- save JSON result to a specified path,
- save text result when not using `--json`,
- overwrite behavior if that is the desired current policy.

**Step 2: Run focused RED tests**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
Expected: FAIL due to missing CLI argument/behavior.

**Implemented:**
- added focused CLI regression tests for `prompt-chain-run --output-file` in text mode,
- added focused CLI regression tests for `prompt-chain-run --output-file --json`,
- updated existing run-mode test args to include the new optional `output_file` field,
- confirmed RED before implementation on missing file-write behavior.

**Verified:**
- `pytest tests/test_prompt_chain_cli.py -q`
- RED confirmed before GREEN on missing output-file behavior.

---

### Task 6: Implement `prompt-chain-run --output-file`

**Status:** completed

**Objective:** Let operators persist run artifacts without scraping stdout.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`
- Optional docs sync: `docs/README-DEV.md`

**Implementation targets:**
- support `--output-file <path>` for `prompt-chain-run`,
- when paired with `--json`, save structured JSON,
- otherwise save the same text artifact the command would print,
- still print a concise confirmation or preserve current stdout behavior intentionally.

**Constraint:**
Do not add multiple format flags unless needed by tests.
Keep the first slice simple.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
- `ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`
- `python -m py_compile cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`

**Implemented:**
- added parser support for `prompt-chain-run --output-file <path>`,
- text mode now saves the same rendered run artifact that would otherwise be printed,
- JSON mode now saves the structured run payload without changing the contract shape,
- file targets are created with parent directories when needed and CLI prints a concise save confirmation instead of duplicating the artifact on stdout.

**Verified:**
- `pytest tests/test_prompt_chain_cli.py -q`
- `python -m py_compile cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`
- `ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`

---

### Task 7: Add GUI result actions for copy/save

**Status:** completed

**Objective:** Make the GUI result surface easier to consume without widening chain semantics.

**Files:**
- Modify: `gui/dialogs/prompt_chains.py`
- Modify: `tests/test_prompt_chain_dialog.py`

**Implementation targets:**
Add small actions for at least:
- copy final output,
- copy final summary,
- save displayed result to file.

**Constraint:**
Keep the UI read-first and high-contrast.
Do not add a complex tabbed workflow builder.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_dialog.py -q`
- `ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py`

**Implemented:**
- added result-header actions for `Copy final output`, `Copy final summary`, and `Save result`,
- stored the latest final output/summary separately from the rendered pane so copy actions target the bounded final artifacts,
- reset cached final-output/final-summary state on `Clear`,
- saved the current rendered plaintext result to a user-selected file without changing chain execution semantics.

**Verified:**
- `pytest tests/test_prompt_chain_dialog.py -q`
- `ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py`

---

## Phase 3 — Model clarity cleanup

### Task 8: Document legacy-vs-active model fields

**Objective:** Mark model drift explicitly before changing code paths.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: `docs/README-DEV.md` if developer guidance needs correction
- Reference: `models/prompt_chain_model.py`

**Implementation targets:**
Document that the active runtime semantics are linear and do not currently depend on:
- `input_template`,
- `condition`,
- `variables_schema`

if those remain compatibility-only.

**Verification:**
- Re-read the changed sections and confirm they no longer imply active workflow semantics for those fields.

---

### Task 9: Add failing model tests for compatibility-only legacy field handling

**Objective:** Lock down the intended narrow semantics before cleanup.

**Files:**
- Modify: `tests/test_prompt_chain_model.py`
- Reference: `models/prompt_chain_model.py`

**Test targets:**
- legacy fields can still round-trip if required for compatibility,
- active defaults do not imply conditional workflow behavior,
- UI-facing defaults remain bounded and linear.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_model.py -q`
Expected: FAIL if current semantics are too broad or under-specified.

---

### Task 10: Implement compatibility-safe model cleanup

**Objective:** Align active model semantics with the bounded runtime while preserving import/storage compatibility where necessary.

**Files:**
- Modify: `models/prompt_chain_model.py`
- Modify: `gui/dialogs/prompt_chain_editor.py` if field presentation needs cleanup
- Modify: `cli/commands.py` if export/show surfaces must label legacy fields differently
- Modify: `tests/test_prompt_chain_model.py`
- Modify: `tests/test_prompt_chain_dialog.py` and/or `tests/test_prompt_chain_cli.py` if affected

**Implementation targets:**
- keep legacy fields only as compatibility boundaries if needed,
- remove or hide active-semantic implications in UI/CLI,
- ensure exported/shown data does not oversell unsupported runtime behavior.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q`
- `ruff check models/prompt_chain_model.py gui/dialogs/prompt_chain_editor.py`

---

## Phase 4 — Supporting note: older durable-history track

### Task 11: Design the smallest bounded recent-run evidence record for any future durable storage

**Objective:** Define the minimum bounded recent-history shape before adding any future durable storage.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Reference: prompt repository/history seams to inspect during implementation

**Record target:**
Persist only bounded evidence such as:
- chain id,
- run timestamp,
- aggregate status,
- input preview,
- final output preview.

**Constraint:**
No analytics-first schema.
No token-heavy archival of every request/response by default.

**Verification:**
- Confirm the documented record shape is clearly bounded and trust/debug oriented.

---

### Task 12: Add failing repository/backend tests for bounded recent chain history

**Objective:** Freeze the minimum bounded recent-history semantics in tests.

**Files:**
- Modify: repository/history tests once seam is identified
- Modify: `tests/test_prompt_chain_backend.py`

**Test targets:**
- successful run records a recent history entry,
- recent history is bounded by limit,
- inspection returns newest-first recent entries,
- no dashboard/analytics behavior is introduced.

**Verification:**
Run focused tests on the touched repository/backend files and confirm RED.

---

### Task 13: Implement lightweight persisted chain-run history

**Objective:** Make recent chain evidence survive beyond the current GUI session.

**Files:**
- Modify: repository/history seam identified during Task 12
- Modify: `core/prompt_manager/chains.py`
- Modify: `cli/commands.py` if a small inspection command is added in the same slice
- Modify: GUI file only if surfaced there

**Implementation targets:**
- record one lightweight history entry per completed run,
- expose bounded recent inspection,
- reuse existing history/repository patterns where practical.

**Constraint:**
Keep this separate from analytics.
Do not build charts, scoring, or broad dashboards in the same slice.

**Verification:**
Run the focused repository/backend tests plus Ruff and py_compile on touched files.

---

## Phase 5 — Final polish and consistency

### Task 14: Unify step-status vs run-status vocabulary

**Objective:** Remove avoidable status ambiguity across core, CLI, and GUI.

**Files:**
- Modify: `core/prompt_manager/chains.py`
- Modify: `cli/commands.py`
- Modify: `gui/dialogs/prompt_chains.py`
- Modify: affected tests
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`

**Target vocabulary:**
- step status: `success | failed | skipped`
- run status: `success | partial | failed`

**Constraint:**
Do not change vocabulary in one surface only.
Update all touched surfaces and tests together.

**Verification:**
Run focused backend/CLI/dialog tests covering status display and JSON/text output.

---

### Task 15: Final docs and verification sweep

**Objective:** Sync docs and prove the final bounded feature state.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-implementation-plan.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-next-slices-plan.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md` only if feature truth changed
- Modify: `docs/CHANGELOG.md` if shipped behavior changed
- Optional: `docs/README-DEV.md`

**Verification commands:**
Run only the relevant focused suite for touched seams, for example:
- `pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py tests/test_prompt_chain_model.py -q`
- `ruff check cli core gui models tests`
- `python -m py_compile cli/parser.py cli/commands.py core/prompt_manager/chains.py gui/dialogs/prompt_chains.py models/prompt_chain_model.py`
- `pyright` on touched files *(do weryfikacji if unavailable in environment)*

**Definition of done:**
- result contract is explicit,
- alias semantics are no longer ambiguous for automation,
- result artifacts are easier to save/consume,
- model drift is reduced or clearly compatibility-scoped,
- lightweight history remains bounded,
- docs reflect shipped truth.

---

## Recommended execution order

If you want the best cost/value order, execute:
1. Task 1–4 (contract tightening)
2. Task 5–7 (artifact handling)
3. Task 8–10 (model cleanup)
4. Task 11–13 (bounded recent history)
5. Task 14–15 (consistency + docs)

---

## Lowest-risk next slice

If only one new slice should be implemented next, choose:

**Task 5–6: save-to-file result artifact support**

Reason:
- canonical result semantics are now explicit,
- the next highest-value gap is persisting run artifacts without scraping stdout,
- this extends result consumption without widening chain semantics.
