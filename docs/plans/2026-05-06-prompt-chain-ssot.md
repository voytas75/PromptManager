# PromptManager — Prompt Chain SSOT

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-05-06 (post-Slice 8 semantics sync; semantic slices queued)
Canonical product SSOT: `docs/product-ssot.md`
Related shipped cycle: `docs/plans/2026-04-29-prompt-chain-clarity-continuity-roadmap.md`
Active implementation ledger: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`
Bounded semantic execution plan: `docs/plans/2026-05-06-prompt-chain-semantic-slices-plan.md`
Supporting execution ledger: `docs/plans/2026-05-06-prompt-chain-implementation-plan.md`

## Purpose

This file is the single source of truth for the bounded PromptManager prompt-chain surface.

It exists to answer five questions:

1. what prompt chains are in PromptManager,
2. what they are not,
3. what behavior is canonical today,
4. what improvements are allowed next,
5. how future implementation slices should be selected.

If chat discussion, old implementation assumptions, or legacy data-model fields conflict with this file, this file wins until explicitly revised.

---

## One-sentence definition

**Prompt chains are a bounded supporting feature for running an ordered sequence of stored prompts over one plain-text input, with optional web-search enrichment and optional final summarization, without turning PromptManager into a workflow engine.**

---

## Product role

Prompt chains are:
- a **supporting execution surface**,
- subordinate to prompt-asset management,
- useful when an operator wants to reuse multiple prompts as one bounded linear run,
- allowed to improve readability, trust, inspectability, and operator control.

Prompt chains are not the product center.
Prompt assets remain the product center.

Short rule:

**prompt assets first, chains second, orchestration never by default**

---

## Canonical current model

The canonical prompt-chain model for planning and implementation is:

- one selected stored chain,
- one plain-text input provided by the operator,
- ordered linear steps,
- each successful step response becomes the next step input,
- optional per-step web-search enrichment before execution,
- optional final summary derived from the last successful step,
- bounded GUI and CLI presentation of the run result.

This means prompt chains are currently a **linear runner with inspectable results**, not a general workflow system.

---

## Canonical operator jobs

Prompt chains should primarily help with these jobs:

1. run a known multi-step prompt recipe over one concrete input,
2. inspect how output moved from one step to the next,
3. compare whether the chain result is useful enough to reuse or refine,
4. rerun a known chain with different input without rebuilding the flow manually,
5. export, inspect, or validate a chain definition as an asset-adjacent operational object.

If a proposed chain feature does not strengthen one of those jobs, it should usually not lead the roadmap.

---

## Canonical in-scope behavior

### 1. Definition and storage
Allowed:
- stored chain records with ordered steps,
- create/edit/delete/import/export of chain definitions,
- bounded metadata that improves human inspection,
- compatibility handling for legacy payload fields when necessary.

### 2. Execution
Allowed:
- one plain-text chain input,
- sequential step execution,
- stop-on-failure vs continue-on-failure,
- optional web-search enrichment,
- optional final summary,
- explicit final raw output and explicit summary when both are available.

### 3. Inspection and trust
Allowed:
- readable GUI chain details,
- readable CLI inspection,
- per-step run evidence,
- structured JSON output for inspection and run results,
- lightweight chain-run history when tied to trust/debug value,
- validation and dry-run checks for definitions before persistence.

### 4. Operator ergonomics
Allowed:
- duplicate chain,
- reorder steps,
- prompt preview while editing,
- export/import round-trip,
- save run result artifacts,
- compact execution metadata that explains what happened.

---

## Explicitly out of scope

Do not expand prompt chains into:
- branching workflows,
- conditional graph execution,
- loops,
- retries as workflow semantics,
- hidden background automation,
- scheduler-first execution,
- multi-agent orchestration,
- a separate chain-first product surface,
- a broad no-code automation builder,
- a replacement for the prompt asset library.

Also avoid:
- adding model complexity that the GUI/CLI cannot explain clearly,
- keeping dead model fields as if they were active product semantics,
- machine-oriented payload growth without operator-visible value.

---

## Strategic rules for future work

When choosing prompt-chain work, prefer this order:

1. **clarity of what the chain is**
2. **inspectability of what happened during a run**
3. **lifecycle completeness of the chain as a manageable object**
4. **operator ergonomics for editing and reuse**
5. **only then small execution-side extensions**

Practical selection rule:
- prefer features that make existing chains easier to understand, inspect, validate, export, or rerun,
- avoid features that mainly add engine power while reducing legibility.

---

## Current confirmed strengths

The following are treated as confirmed current behavior:
- GUI chain CRUD exists,
- CLI chain list/show/apply/run/validate/export exists,
- `prompt-chain-show --json` exists,
- `prompt-chain-run --json` exists,
- chain execution uses a single plain-text input,
- later steps receive previous step output,
- optional web-search enrichment exists,
- optional summary exists,
- run results now distinguish `step_outputs`, `final_output_text`, and `final_summary_text`,
- per-step run metadata now includes operator-facing `step_label` and machine-facing `step_output_key`,
- GUI/CLI already contain bounded readability cues from the completed clarity cycle.

These should not be replanned as new product work unless code/tests prove regression.

---

## Canonical result-semantics rule

Prompt-chain result surfaces should distinguish clearly between:
- operator-facing labels,
- machine-facing stable keys,
- final chain outcome surfaces.

Current confirmed semantics:
- each step has a human-facing `step_label`,
- each step has a stable machine-facing `step_output_key`,
- run results expose `final_output_text` and `final_summary_text` separately,
- `step_outputs` remains machine-readable and must stay deterministic enough for script/API consumption,
- `step_outputs` now uses canonical machine keys only,
- alias semantics are exposed explicitly via `step_aliases`,
- the terminal result now exposes `final_step_id`, `final_step_output_key`, and `final_step_label`.

Current implementation-compatible rule:
- `step_{n}` is the stable machine key for a step output,
- `output_variable` remains an operator-facing alias and must not replace the canonical machine key in automation,
- GUI/CLI may present both when that improves clarity,
- automation should consume canonical machine keys and explicit alias metadata, not infer alias equality from duplicated outputs.

Current preferred direction:
- keep one clearly canonical machine-oriented output identity,
- keep aliases explicit rather than implicit,
- avoid silent ambiguity about which key should be consumed by automation.

Short rule:

**human label for operators, stable output key for machines, no hidden ambiguity about canonical output identity**

---

## Current confirmed gaps

The following are the main canonical gaps after the shipped contract-tightening and CLI artifact-save slices:

### 1. Result-surface completeness gap
The core result contract is explicit and run artifacts can now be saved directly, but bounded consumption still needs easier selective output modes and stable handoff ergonomics, especially for:
- compact CLI result modes,
- tighter export consistency,
- small operator shortcuts that reduce manual post-processing.

Canonical direction:
- keep the single structured contract,
- keep artifact handling bounded and predictable,
- avoid introducing competing payload shapes.

### 2. Result-consumption ergonomics gap
The product now distinguishes:
- final raw output,
- final summary,
- per-step outputs,
- per-step label vs machine key,
- canonical machine outputs vs explicit aliases,
- final terminal step identity.

The remaining gap is bounded result consumption ergonomics, for example:
- compact CLI output modes,
- tighter JSON/export stability,
- small handoff conveniences around the saved artifact flow.

### 3. Model clarity gap
The persisted chain dataclasses still contain fields that may outlive the active bounded runtime semantics.

Primary drift candidates:
- `PromptChainStep.input_template`
- `PromptChainStep.condition`
- `PromptChain.variables_schema`

Canonical direction:
- active runtime semantics should drive the product narrative,
- legacy fields may remain at import/storage boundaries when needed,
- but they should not continue to imply a broader workflow model than the current product actually supports.

### 4. Editor ergonomics gap
The chain editor should become easier to use without becoming a builder platform.

Allowed next improvements:
- duplicate,
- reorder,
- better prompt previews,
- clearer warnings and validation.

### 5. Run-semantics consistency gap
The product now exposes richer run-result fields, but the feature still has a confirmed semantics gap around:
- aggregate run status,
- `final_step_*` versus terminal execution step identity,
- consistent consumption of those semantics across backend, CLI, GUI, and history surfaces.

Canonical direction:
- compute one backend-owned aggregate `run_status`,
- distinguish clearly between the last successful step that produced final output and the terminal step where execution ended,
- avoid local CLI/GUI heuristics that reinterpret the same run differently.

### 6. Bounded history gap
Lightweight run history now exists as a bounded backend seam, but its product semantics are not fully unified yet.
The remaining gap is:
- consistent status semantics between backend and GUI recent-run views,
- clear positioning of current history as bounded trust/debug evidence,
- explicit confirmation whether the current surface is session/process-level or durable across restarts.

It should remain tied to evidence and reuse value, not expand into dashboard-first behavior.

---

## Approved next subfunctions

The following subfunctions are explicitly approved as next-step planning candidates:

### Priority A
- explicit result-contract tightening for canonical output-key vs alias semantics
- explicit final-step contract fields when justified
- `prompt-chain-run --final-output-only`
- `prompt-chain-run --summary-only`
- `prompt-chain-run --status-only`

### Priority B
- save run result artifact to file
- tighter deterministic JSON/export contract
- bounded model-clarity cleanup for legacy-semantics fields
- GUI duplicate-chain action
- GUI step reorder controls
- bounded recent-runs / lightweight chain history
- minimum durable chain-run evidence record

### Approved semantic follow-up
- backend-owned `run_status` contract
- explicit `terminal_step_*` fields when justified to avoid collision with `final_step_*`
- CLI/GUI/history alignment on one run-semantics contract

---

## Canonical run-status and finality direction

The next bounded semantic cleanup should distinguish between:
- overall chain run status,
- final output-producing step identity,
- terminal execution step identity.

Preferred direction:
- `final_step_*` should mean the last successful step that produced the final usable output,
- terminal-step fields should mean the last executed step where the chain actually ended,
- aggregate `run_status` should be computed once in backend and consumed by CLI/GUI/history instead of being re-derived locally.

This semantic tightening is implementation-approved and tracked in:
- `docs/plans/2026-05-06-prompt-chain-semantic-slices-plan.md`

Short rule:

**one backend-owned run status, explicit final-vs-terminal step semantics, no per-surface reinterpretation**

---

## Minimum durable chain-run evidence record

If lightweight chain-run history is persisted, the durable record must stay bounded and evidence-first.

**Required fields:**
- `chain_id`
- `chain_name` snapshot or another resolvable chain identity
- `run_timestamp`
- aggregate `status`
- bounded `input_preview`
- bounded `final_output_preview`
- `final_step_output_key`

**Optional only when already cheaply available:**
- `final_step_label`
- `final_step_id`

**Do not persist by default:**
- full request blobs,
- full response blobs,
- per-step raw payload archives,
- analytics-first aggregates,
- dashboard-only metadata.

**Policy:**
- newest-first retrieval is allowed,
- retention must stay bounded,
- the record exists for trust/debug/reuse only,
- this is not a reporting or observability subsystem.

---

## Model policy

For future implementation work, treat this as the preferred model direction:

- active prompt-chain semantics should be represented by the minimum fields needed for:
  - ordered execution,
  - prompt reference,
  - failure behavior,
  - bounded operator-visible metadata.
- legacy fields may be accepted at import boundaries if required for backward compatibility,
  but they should not continue to drive the product narrative if the runtime no longer uses them.

Short rule:

**active model semantics must match the active UX/runtime, not historical capability leftovers**

---

## CLI policy

Prompt-chain CLI should support the full bounded lifecycle:
- list,
- show,
- validate,
- apply/import,
- export,
- run,
- structured JSON reads/results.

CLI is an operator surface, not only an internal dev seam.
That means readability and machine-readable output are both first-class requirements.

---

## GUI policy

Prompt-chain GUI should stay:
- simple,
- high-contrast,
- read-first,
- explicit about chain input, per-step input/output, and terminal outcome.

Do not turn the GUI into a visual workflow canvas.
Prefer compact form-and-list editing over diagram-builder behavior.

---

## Planning rule

Use this file as the feature-level SSOT when planning Prompt chain work.

When writing implementation plans:
1. start from the confirmed gaps and approved subfunctions in this file,
2. keep slices small and bounded,
3. prefer one seam at a time,
4. update the plan after every implemented slice,
5. revise this SSOT only when the feature truth changes.

Recommended planning order:
1. model clarity,
2. lifecycle completeness,
3. result semantics,
4. debug metadata,
5. editor ergonomics,
6. bounded history.

---

## Definition of done for future prompt-chain slices

A prompt-chain slice is done only when:
- the scope stays inside this bounded feature definition,
- GUI/CLI/core semantics remain aligned,
- the change increases clarity, inspectability, lifecycle completeness, or bounded ergonomics,
- focused tests prove the seam,
- docs/plan state is updated.

If a candidate slice mostly adds engine complexity without clear operator trust value, it is out of scope by default.
