# PromptManager Detail Fork Handoff Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce hesitation between refining and forking by adding one bounded fork-specific next-step cue on the existing prompt detail handoff surface, without reopening controller logic, workspace flow, or broader decision-model semantics.

**Architecture:** This roadmap stays on one widget-local inspect/reuse seam. It targets the already-existing detail handoff surface in `gui/widgets/prompt_detail_widget.py`, especially the relationship between `promptDecisionSummary`, `promptNextActionSummary`, and `promptWorkspaceHandoffCue`. The slice should stay mechanical: preserve current decision and next-action strings, add only the missing fork-specific handoff mapping, and verify it through focused widget tests rather than controller refactors.

**Tech Stack:** Python 3.13, PySide6 detail widget, pytest, Ruff, PromptManager docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-prompt-chains-typing-cleanup-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/product-ssot.md` keeps inspection clarity and reuse/refinement guidance in the core product loop.
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` already frames detail refine/fork action clarity as a valid bounded direction, but the earlier slice is marked delivered previous work and should not be reopened wholesale.
- `gui/widgets/prompt_detail_widget.py` already has a dedicated handoff surface:
  - `promptDecisionSummary`
  - `promptNextActionSummary`
  - `promptWorkspaceHandoffCue`
- focused tests already prove widget-local handoff coverage for:
  - `Validate before reuse` -> `Open in Workspace before validating reuse.`
  - `Edit Prompt before reuse` -> `Edit Prompt before reuse.`
- current code and tests also confirm a distinct fork-oriented action string exists on the decision side:
  - `Fork before editing`
- no focused fork-specific workspace handoff cue has yet been confirmed in widget tests.

So the next useful move is:

**detail handoff clarity -> add one fork-specific next-step cue on the existing widget surface**

---

## Scope guardrails

This roadmap may improve only:
- widget-local handoff wording/mapping in `gui/widgets/prompt_detail_widget.py`,
- focused widget tests in `tests/test_prompt_detail_widget.py`,
- docs pointers for the bounded slice after it lands.

This roadmap must not introduce:
- new buttons or actions,
- prompt controller rewrites,
- workspace workflow changes,
- persistence/model changes,
- broader refine/fork heuristics or prompt-action policy changes.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- detail decision summary and next-action summary labels are already shipped,
- validation-first handoff cues are already shipped,
- edit-before-reuse handoff cue is already shipped,
- typing cleanup slices C3.1-C3.4 are already shipped and should stay out of scope here.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> what is the smallest handoff-clarity slice that makes fork-first guidance as explicit as the existing validation-first and edit-first cues?

Preferred answer shape:
- one widget file,
- one focused test file,
- no new actions,
- one concrete `Next step:` cue.

---

## Stage A — Detail fork handoff clarity

### Task 1: Audit the fork-specific handoff gap

**Status:** completed

**Objective:** Confirm the exact decision / next-action string contract for the fork path and verify the missing widget-local cue.

**Files:**
- Inspect: `gui/widgets/prompt_detail_widget.py`
- Inspect: `tests/test_prompt_detail_widget.py`
- Inspect: `gui/workspace_history_controller.py`
- Maintain: `docs/plans/2026-05-10-detail-fork-handoff-clarity-roadmap.md`

**Audit checks:**
- confirm the current fork-oriented next-action string (`Fork before editing` or a more specific successor string),
- confirm whether `promptWorkspaceHandoffCue` currently renders a fork-specific handoff line,
- keep the slice bounded to the widget surface even if controller strings are reused.

**Exit criterion:**
One exact fork-path string contract and one confirmed missing handoff cue.

---

### Task 2: Reproduce the gap with a focused RED widget test

**Status:** completed

**Objective:** Add one failing test proving the detail widget does not yet show the fork-specific `Next step:` cue.

**Files:**
- Modify: `tests/test_prompt_detail_widget.py`
- Verify: `gui/widgets/prompt_detail_widget.py`

**RED target shape:**
- `display_prompt(prompt)`
- `update_decision_summary("Refine before reuse")` or the current fork-oriented decision if needed
- `update_next_action_summary("Fork before editing")`
- assert `promptWorkspaceHandoffCue` becomes visible and contains a bounded fork-specific `Next step:` message

**Exit criterion:**
One focused failing widget test for the fork handoff gap.

---

### Task 3: GREEN minimal widget-local handoff fix

**Status:** completed

**Objective:** Add the smallest fork-specific handoff mapping on the existing detail cue surface.

**Files:**
- Modify: `gui/widgets/prompt_detail_widget.py`
- Verify: `tests/test_prompt_detail_widget.py`

**Implementation target:**
- reuse the existing workspace handoff cue mechanism,
- add one mapping for the fork-oriented next-action string,
- keep wording concrete and action-first,
- do not change any controller decision logic.

**Exit criterion:**
Focused test passes and no existing detail cue behavior regresses.

---

### Task 4: Verify non-regression and sync pointer-chain

**Status:** completed

**Objective:** Verify the seam and keep next-cycle docs unambiguous after the bounded slice lands.

**Files:**
- Verify: `tests/test_prompt_detail_widget.py`
- Modify: this file
- Modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Verification pack:**
```bash
pytest tests/test_prompt_detail_widget.py -q
ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py
```

**Exit criterion:**
Fork-handoff widget seam is green and docs identify the most recently delivered bounded slice correctly.

---

## Current recommended next slice

This roadmap is the active bounded successor only until the fork-specific handoff cue is delivered.

If the fork cue lands cleanly, do not leave this file acting like an open-ended detail-action umbrella.
A later successor must be chosen explicitly.

---

## Closure / successor rule

After this delivery:
- do not reopen the shipped fork handoff wording seam itself unless a focused regression appears,
- do not expand this slice into controller or workspace logic,
- choose another detail-action successor only if it stays on the same existing cue surface,
- keep prompt detail guidance subordinate to the broader prompt-asset loop.

---

## Success definition

This slice is successful if:
- the detail widget shows one explicit fork-specific `Next step:` cue,
- existing validation/edit handoff cues remain intact,
- prompt detail widget tests stay green,
- docs clearly show this slice as delivered rather than leaving it implied or active forever.
