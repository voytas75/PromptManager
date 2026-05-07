# PromptManager — Prompt Chain Next Slices Plan

Status: supporting note
Owner: Wojtek / Prompt Manager Team
Updated: 2026-05-06 (post-Slice 8 SSOT sync; superseded as main next-step guide by rollout plan)
Feature SSOT: `docs/plans/2026-05-06-prompt-chain-ssot.md`
Execution ledger: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`
Canonical product SSOT: `docs/product-ssot.md`

## Purpose

This file is the short forward-looking plan for the next prompt-chain slices after the completed clarity and step-semantics work.

It is not the main execution ledger.
This file remains a forward-looking supporting note only.
The historical implementation truth for earlier slices lives in `docs/plans/2026-05-06-prompt-chain-implementation-plan.md`.
The active implementation truth now lives in `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`.

---

## Confirmed current position

Treat the following as already delivered unless focused regression proves otherwise:
- bounded linear prompt-chain execution,
- single plain-text chain input,
- step-to-step handoff visibility,
- explicit `final_output_text` vs `final_summary_text`,
- operator-facing `step_label`,
- machine-facing `step_output_key`,
- CLI `show/run/validate/export` lifecycle,
- GUI bounded chain CRUD and run inspection.

This means the next slices should not add more engine power first.
They should tighten contracts, consumption, and bounded operator ergonomics.

---

## Strategic next-slice order

### 1. Result contract tightening

**Why first:**
The main remaining ambiguity is not execution power. It is the contract around:
- canonical output identity,
- alias semantics,
- terminal result identity.

**Desired outcome:**
- one explicit machine-oriented canonical output identity,
- no hidden ambiguity between `step_output_key` and `output_variable`,
- clearer final-step contract for automation and operator trust.

**Likely seams:**
- `core/prompt_manager/chains.py`
- `cli/commands.py`
- `docs/plans/2026-05-06-prompt-chain-ssot.md`
- focused backend/CLI tests

**Good bounded slice shapes:**
- explicit canonical-vs-alias output semantics
- explicit final-step contract fields
- deterministic JSON contract tightening

---

### 2. Result-surface completeness

**Why second:**
Once the contract is explicit, the next highest-value work is making result consumption easier without adding engine complexity.

**Desired outcome:**
- short CLI result modes,
- stable save/export paths,
- cleaner script/operator consumption.

**Good bounded slice shapes:**
- `prompt-chain-run --final-output-only`
- `prompt-chain-run --summary-only`
- `prompt-chain-run --status-only`
- save run result artifact to file

**Likely seams:**
- `cli/parser.py`
- `cli/commands.py`
- focused CLI tests

---

### 3. Model-clarity cleanup

**Why third:**
The feature is now easier to inspect, so the next structural debt is model drift between active runtime semantics and persisted fields.

**Primary candidates:**
- `PromptChainStep.input_template`
- `PromptChainStep.condition`
- `PromptChain.variables_schema`

**Desired outcome:**
- active semantics match active runtime,
- legacy fields stay compatibility-only when needed,
- docs and payloads stop implying broader workflow behavior than the product actually supports.

**Likely seams:**
- `models/prompt_chain_model.py`
- `gui/dialogs/prompt_chain_editor.py`
- `cli/commands.py`
- focused model/dialog/CLI tests

---

### 4. Editor ergonomics

**Why after contract/consumption/model work:**
This improves daily use, but it should follow the semantic tightening so the editor does not harden old ambiguity.

**Good bounded slice shapes:**
- duplicate chain
- reorder steps
- better prompt preview
- clearer warnings/validation

**Constraint:**
Keep the GUI form-first and read-first.
Do not move toward a visual workflow builder.

---

### 5. Bounded history

**Why last:**
Useful, but lower priority than contract, result consumption, and model clarity.

**Desired outcome:**
- lightweight recent-runs evidence,
- trust/debug support,
- no dashboard-first expansion.

---

## Current recommended next slice

### Recommended next slice
**Prompt-chain canonical output contract v1**

**Reason:**
This is now the best next move because the main remaining trust gap is semantic, not executional.
The product already has bounded result-consumption shortcuts, but the machine contract still leaves ambiguity between canonical output keys and operator aliases.

**Scope:**
- explicit canonical machine key semantics for `step_outputs`
- explicit alias semantics separate from canonical machine outputs
- explicit final-step contract fields in structured run results

**Why this before further result-surface work:**
- it removes the highest remaining ambiguity for automation,
- it stabilizes the JSON contract before adding more artifact/export conveniences,
- it keeps the feature inside the bounded linear runner model,
- it reduces the risk of hardening mixed alias semantics across CLI/GUI/docs.

**Constraint:**
Do not redesign the full execution engine or add a second competing payload shape.
Keep this slice to deterministic contract tightening only.

---

## Deferred but approved follow-up candidates

After the recommended next slice, prefer this order:
1. canonical output-contract tightening,
2. save-to-file result artifact support,
3. bounded model-clarity cleanup,
4. GUI duplicate/reorder ergonomics,
5. lightweight run history.

---

## Anti-scope

Do not use the next slices to introduce:
- branching workflows,
- loop/retry workflow semantics,
- scheduler-first behavior,
- multi-agent orchestration,
- canvas-style chain editing,
- payload growth without operator-visible value,
- broad automation that outruns the product model.

Short rule:

**tighten the bounded linear runner; do not evolve it into a workflow engine by accident**
