# PromptManager Next Cycle Closure and Next Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Formalnie domknąć cykl roadmapowy z 2026-04-26 i przygotować następny bounded execution track skupiony na asset-to-run-to-refine coherence bez łamania asset-first SSOT.

**Architecture:** Ten plan nie zmienia produktu ani nie resetuje wcześniejszych roadmap. Najpierw stabilizuje granicę „cykl ukończony” w docs, potem definiuje nowy mały tor wykonawczy oparty o istniejące inspect/detail, execution history, workspace handoff i CLI parity seams. Priorytetem jest legibility operator path, nie nowy workflow, panel ani persistence model.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, PromptManager docs under `docs/`, existing inspect/detail + execution-history + headless seams.

---

## Why this plan exists now

Live repo review confirms:
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` is fully implemented in scope,
- `docs/plans/2026-04-26-next-cycle-roadmap.md` has Slice 1–7 marked `implemented`,
- `docs/product-ssot.md` remains aligned with the shipped asset-first / trustworthy-runs posture,
- `docs/CHANGELOG.md` already records the delivered evaluation/governance cues,
- git worktree is clean at `f9439ea` on `master`.

So the next useful move is not another opportunistic micro-slice inside the old cycle, but:
1. mark the cycle boundary explicitly,
2. define the next bounded track,
3. execute the next slice against that new boundary.

---

## Current confirmed boundary

The finished 2026-04-26 cycle already covers:
- evidence sufficiency fallback,
- missing-evidence reason split,
- decision provenance cues,
- replace-path `Keep baseline`,
- validation freshness,
- CLI parity for decision/next/freshness,
- limited-evidence provenance.

That means the next plan should **not** repeat evaluation/governance groundwork.
It should move forward into:

> **asset -> run -> review -> next operator step**

using existing surfaces only.

---

## New execution focus

### Focus area
**Asset-to-run-to-refine coherence v2**

### Product intent
Improve the continuity between:
- selecting/inspecting a prompt asset,
- seeing the latest run evidence,
- understanding whether the evidence is still actionable,
- taking the next bounded operator step in workspace/reuse/refine flow.

### Constraints
Do not add:
- a new review queue,
- a new analytics dashboard,
- a new orchestration canvas,
- background scheduling,
- new persistence only for guidance,
- a CLI-only shadow model.

Reuse first:
- `gui/workspace_history_controller.py`
- `gui/widgets/prompt_detail_widget.py`
- `core/history_tracker.py`
- `cli/commands.py`
- existing tests around workspace history, prompt detail, and main entry.

---

## Recommended next bounded slices

These are ordered by preference.

### Slice A1 — Validation recency next-step cue

**Status:** implemented

**Intent:** If the latest run exists but its freshness is stale, make the next operator step more explicit without changing the decision model.

**Good shape:**
- reuse existing freshness + next-action seams,
- no extra label,
- one compact wording refinement only when freshness is stale,
- preserve stronger compare / baseline decisions.

**Implemented:**
- Reused the existing inspect/detail `Recommended next action` seam instead of adding a new review/status surface.
- Single-run prompts that still look reusable but only expose stale validation evidence now tighten the next step to `Validate before reuse`.
- Kept the decision model unchanged (`Reuse as-is`) and left stronger compare / baseline / missing-baseline paths untouched.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_missing_evidence_reason_for_single_run -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- result: `53 passed`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py`
- result: `All checks passed!`

### Slice A2 — Workspace handoff coherence after stale evidence

**Intent:** Make the path from stale latest evidence to workspace validation more legible when the user opens a prompt in workspace.

**Good shape:**
- reuse existing non-executing workspace handoff message,
- add only one bounded stale-evidence-aware hint when such evidence already exists,
- do not auto-run anything,
- do not create a separate review flow.

**Candidate behavior:**
- generic handoff remains for prompts without stale-evidence context,
- stale-evidence prompts strengthen the existing validation hint instead of branching into a new workflow.

**Primary files:**
- existing prompt-actions/workspace handoff controller seam
- related handoff tests
- `docs/CHANGELOG.md`

### Slice A3 — Result-to-next-step wording refinement for one-run prompts

**Intent:** Improve operator coherence for the common path where only one run exists and the system already knows evidence is thin.

**Good shape:**
- reuse missing-evidence wording,
- no new status surface,
- make the next action more directly tied to validation/refinement,
- keep `Decision` conservative.

**Candidate behavior:**
- `Decision` remains `Reuse as-is` or other existing safe fallback,
- `Recommended next action` becomes slightly more operationally explicit than a pure evidence statement if and only if that improves the handoff.

**Primary files:**
- `gui/workspace_history_controller.py`
- `tests/test_workspace_history_controller.py`
- `tests/test_prompt_detail_widget.py`

### Slice A4 — Headless parity guard for the next handoff cue

**Intent:** After one GUI/shared wording refinement lands, decide whether the same bounded semantic belongs in `history-analytics`.

**Rule:**
- If the new wording comes from already-shared fields, extend CLI parity with tests.
- If it is purely widget-local or workspace-action-local, keep CLI unchanged and document why.

**Primary files:**
- `cli/commands.py` only if the changed wording is shared
- `core/history_tracker.py` only if a shared field must be added
- `tests/test_main_entry.py`

---

## Recommended first slice

### Pick first: Slice A1 — Validation recency next-step cue

Why this first:
- directly fits the unfinished `asset-to-run-to-refine coherence` goal,
- uses seams already proven in `WorkspaceHistoryController`,
- has an obvious RED test path,
- does not need new persistence,
- can stay entirely bounded to inspect/detail semantics.

Why not start with A2 first:
- workspace handoff is a secondary seam and easier to over-specialize,
- inspect/detail is still the more canonical decision-support surface.

---

## Implementation brief for the first slice

### Task 1: Close the previous cycle in docs

**Objective:** Explicitly mark the 2026-04-26 roadmap as complete in practice and point to this follow-up plan as the next execution boundary.

**Files:**
- Modify: `docs/plans/2026-04-26-next-cycle-roadmap.md`
- Modify: `docs/CHANGELOG.md` only if a user-visible closure note is warranted
- Reference: `docs/product-ssot.md`

**Steps:**
1. Add a short note near the roadmap intro or end that the cycle is complete in its planned scope.
2. Point the next execution track at this new plan file.
3. Do not rewrite product SSOT unless product posture changed.

**Verification:**
- `git diff -- docs/plans/2026-04-26-next-cycle-roadmap.md`
- Review wording for consistency with `docs/product-ssot.md`

### Task 2: Write failing tests for stale-validation next action

**Objective:** Prove the current inspect/detail seam does not yet provide the desired stale-evidence guidance.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`
- Maybe modify: `tests/test_prompt_detail_widget.py` if render behavior needs locking too

**Test target:**
Add one focused scenario where:
- latest run exists,
- freshness is `stale`,
- there is not enough stronger compare evidence to override the path,
- current next-action wording remains too generic.

**Good RED shape:**
- assert current/future desired wording exactly,
- confirm test fails before implementation for the expected reason.

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q`
- Expected before implementation: one failing assertion on next-action wording

### Task 3: Implement minimal stale-validation next-step refinement

**Objective:** Tighten the existing next-action seam for stale-evidence cases without changing unrelated decision logic.

**Files:**
- Modify: `gui/workspace_history_controller.py`

**Implementation rule:**
- prefer one small helper rather than widening `_build_decision_summary()`,
- keep stronger missing-evidence / compare / baseline outcomes intact,
- only refine the next-step wording when stale freshness materially changes the operator posture.

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q`
- Expected after implementation: target test passes

### Task 4: Lock shared/detail parity for the new wording

**Objective:** Ensure the bounded wording appears consistently on the shared/template detail surfaces when applicable.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`
- Modify: `tests/test_prompt_detail_widget.py` only if widget-specific rendering/reset behavior changes

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`

### Task 5: Decide on headless parity

**Objective:** Verify whether the new semantic should surface in `history-analytics`.

**Decision rule:**
- If wording is produced from an already shared field, add a focused `tests/test_main_entry.py` assertion.
- If not, explicitly keep CLI unchanged and record that as deliberate.

**Files:**
- Maybe modify: `tests/test_main_entry.py`
- Maybe modify: `cli/commands.py`
- Maybe modify: `core/history_tracker.py`

**Verification:**
- `.venv/bin/pytest tests/test_main_entry.py -q` if touched

### Task 6: Sync docs after green

**Objective:** Keep the roadmap ledger accurate after the first new slice lands.

**Files:**
- Modify: this file
- Modify: `docs/CHANGELOG.md`
- Modify: `docs/plans/2026-04-26-next-cycle-roadmap.md` only for boundary/forward-pointer wording, not as the main ledger for the new slice

**Required notes:**
- what changed,
- why it stayed bounded,
- exact verification commands/results,
- whether CLI parity changed or intentionally did not.

---

## Done criteria for this plan

This plan is doing its job when:
- the old cycle is explicitly closed,
- the next bounded track is clearly defined,
- the first new slice has a concrete RED→GREEN path,
- the next work stays inside asset-first SSOT,
- no one has to guess whether the next priority is docs cleanup, Pyright debt, or product guidance work.
