# Prompt Chain Run Handoff Clarity v1 — Implementation Brief

Status: proposed

Related roadmap: `docs/plans/2026-04-29-prompt-chain-clarity-continuity-roadmap.md`

## Goal
Make the existing prompt-chain run result easier to interpret by explicitly showing the handoff between steps, so the operator can see that the initial chain input feeds step 1 and each successful step response feeds the next step.

## Why this slice
Live repo review shows the current chain run UI already renders:
- `Input to chain`,
- `Chain outputs`,
- per-step status,
- step 1 input,
- step outputs,
- optional reasoning summary,
- optional chain summary.

However, the current presentation still leaves one continuity gap:
- only the **first** step explicitly shows its input,
- later steps show output, but not clearly the fact that their input came from the previous step,
- operators may need to infer the chain handoff model instead of reading it directly.

This is a good bounded slice because it improves trust without changing execution logic, storage, or chain power.

## Confirmed baseline
From current code and tests:
- backend already runs chains linearly via `previous_response` in `core/prompt_manager/chains.py`,
- `PromptChainRunResult` already carries `chain_input`, `outputs`, `steps`, and optional `summary`,
- GUI result rendering lives in `gui/dialogs/prompt_chains.py::_display_run_result()`,
- current tests in `tests/test_prompt_chain_dialog.py` already verify chain input, outputs, summary rendering, markdown behavior, reasoning display, and last-step summary gating.

So this slice likely does **not** need a backend model change unless recon reveals the GUI cannot distinguish handoff state from existing request/response text.

## Product boundary
Do not add:
- branching or condition redesign,
- new runtime metadata storage,
- richer chain DSL/schema,
- analytics/history expansion,
- CLI redesign in this slice unless the GUI seam proves insufficient.

Prefer:
- a GUI-local continuity cue in the existing run result rendering,
- focused tests in `tests/test_prompt_chain_dialog.py`,
- backend untouched unless a tiny helper is genuinely needed.

## Proposed approach
Use the existing `request_text` / `response_text` already available inside each `PromptChainStepRun` outcome and change only the run-result presentation.

Recommended behavior:
1. Step 1 keeps the current explicit input section.
2. Step 2+ should also render an explicit step-input block.
3. The label for step 2+ should make the handoff obvious, for example one of these bounded forms:
   - `Input from previous step:`
   - `Input to step (from previous step output):`
   - `Received from step N-1:`
4. Keep the step output section as-is unless the final-step reasoning-summary rule suppresses it.
5. Do not duplicate engine explanations elsewhere; let the per-step section carry the continuity cue.

Preferred copy direction:
- calm and literal,
- no orchestration-heavy wording,
- no hidden semantics beyond the already-true linear pipe behavior.

## Likely files to change
Primary target:
- `gui/dialogs/prompt_chains.py`

Likely tests:
- `tests/test_prompt_chain_dialog.py`

Optional only if genuinely needed:
- `core/prompt_manager/chains.py`

## Suggested TDD plan

### 1. Add a failing dialog test for later-step handoff visibility
Add one focused regression in `tests/test_prompt_chain_dialog.py` that:
- creates a two-step chain result,
- sets step 1 request/response and step 2 request/response,
- calls `panel._display_run_result(result)`,
- asserts the plain text clearly shows later-step input / previous-step handoff,
- asserts step 2 input text is visible in output.

Good acceptance shape:
- the rendered plain text contains two step-input sections, not just one,
- the second step input label is continuity-aware,
- the second step input text matches the expected handoff content.

### 2. Run the targeted test to confirm RED
Suggested command:
```bash
pytest tests/test_prompt_chain_dialog.py -k handoff -v
```

If the new test name does not use `handoff`, run the exact test node instead.

### 3. Implement the smallest GUI rendering change
In `gui/dialogs/prompt_chains.py::_display_run_result()`:
- stop limiting step-input rendering to `index == 0`,
- preserve special wording for step 1 if useful,
- introduce a bounded label for later steps so the operator sees the handoff source clearly,
- keep current reasoning-summary and output suppression behavior intact.

### 4. Run focused chain dialog coverage
Suggested command:
```bash
pytest tests/test_prompt_chain_dialog.py -k "prompt_chain_dialog and (handoff or runs_chain or renders_chain_summary or only_last_step_has_reasoning_summary)" -v
```

If selection becomes awkward, run the exact touched test nodes plus the nearby chain dialog tests.

### 5. Run one backend smoke if the implementation touched shared behavior
Only if backend code changes:
```bash
pytest tests/test_prompt_chain_backend.py -v
```

### 6. Docs sync after GREEN
Update:
- `docs/plans/2026-04-29-prompt-chain-clarity-continuity-roadmap.md`
- `docs/CHANGELOG.md`

Expected roadmap update:
- mark `Slice 1 — Prompt chain run handoff clarity v1` as done,
- note that the slice stayed on the run-result presentation seam,
- record whether the change was GUI-only.

## Acceptance criteria
This slice is complete if:
1. A chain run with at least two successful steps clearly shows the later step input in the rendered result.
2. The operator can understand from the UI that later-step input came from earlier chain output.
3. Existing summary/reasoning behavior still works.
4. No engine/storage/schema behavior changed.
5. The roadmap and changelog are updated after tests are green.

## Risks / watchouts
- Avoid overexplaining the flow with too much prose; one explicit per-step cue is enough.
- Do not break the current last-step reasoning-summary logic.
- Do not accidentally duplicate huge text blocks in a way that makes results noisier than before.
- If later-step request text already includes web-search enrichment, render the truth of what was executed, not an invented simplified handoff.

## Verification checklist
- targeted new RED test fails first,
- updated test passes after the patch,
- nearby chain dialog tests remain green,
- docs updated only after code/tests are green,
- no scope drift into CLI/backend unless evidence demands it.

## Recommended next action
Implement this as a **GUI-first bounded slice** on `gui/dialogs/prompt_chains.py` with one new focused dialog regression before touching any production code.
