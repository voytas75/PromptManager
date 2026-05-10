# PromptManager Workspace One-Run Action Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce hesitation on reuse decisions by making one bounded single-run evidence path more actionable on the existing workspace history decision-support seam, without reopening delivered compare-readiness wording, prompt-detail handoff work, or broader history UX.

**Architecture:** This roadmap stays inside the existing trust-surface seam in `gui/workspace_history_controller.py`, specifically where recent execution evidence is converted into compact next-action guidance for reuse/readiness. The slice must remain bounded: audit the current one-run contract, reproduce the wording gap with one focused test, land one compact next-action clarification, and avoid any change to run-summary structure, provenance logic, freshness wording, or unrelated evidence branches.

**Tech Stack:** Python 3.13, PySide6 workspace/history controller, pytest, Ruff, PromptManager docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-workspace-compare-duration-clarity-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/product-ssot.md` keeps trust surfaces and bounded decision support inside the prompt-asset loop.
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` says the next slice should improve operator confidence more than feature breadth.
- `gui/workspace_history_controller.py` already emits compact evidence and next-action strings for limited single-run and compare-readiness blockers, including:
  - `Evidence: only one run available`
  - `Run one more time before reusing`
  - `Validate before reuse`
  - `Run a different prompt version before comparing`
  - `Add ratings to both runs before comparing`
  - `Run both versions again before comparing`
- `tests/test_workspace_history_controller.py` already covers the recent single-run path, so the next safe move is to extend the existing controller test surface rather than introduce a new product seam.

So the next useful move is:

**workspace trust surface -> make the recent one-run next action more explicit without broadening history UX**

---

## Scope guardrails

This roadmap may improve only:
- single-run next-action wording/mapping inside `gui/workspace_history_controller.py`,
- focused controller tests in `tests/test_workspace_history_controller.py`,
- docs pointers for the bounded slice after it lands.

This roadmap must not introduce:
- new widgets,
- prompt-detail handoff rewrites,
- history model/persistence changes,
- broad controller refactors,
- changes to compare-readiness wording or stale-validation behavior unless required by a focused regression.

---

## Main product question for this roadmap

> what is the smallest next-action clarification that makes the recent single-run path more actionable without expanding the workspace-history model?

Preferred answer shape:
- one controller file,
- one focused test file,
- one compact evidence/next-action clarification,
- no new UI surface.

---

## Stage A — Recent one-run action clarity

### Task 1: Audit the current recent one-run contract

**Status:** completed

**Objective:** Confirm the exact current evidence and next-action strings for the recent single-run path and identify the smallest wording gap.

**Files:**
- Inspect: `gui/workspace_history_controller.py`
- Inspect: `tests/test_workspace_history_controller.py`
- Maintain: `docs/plans/2026-05-10-workspace-one-run-action-clarity-roadmap.md`

**Audit checks:**
- confirm where `Evidence: only one run available` is emitted,
- confirm what next action currently maps from that evidence in the recent-freshness path,
- confirm whether the current wording repeats evidence instead of giving an action.

**Exit criterion:**
One exact current string contract and one bounded clarification target.

---

### Task 2: Reproduce the gap with a focused RED controller test

**Status:** completed

**Objective:** Add one failing test that proves the current recent single-run guidance is less actionable than desired.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`
- Verify: `gui/workspace_history_controller.py`

**RED target shape:**
- construct the existing recent single-run history scenario,
- assert current decision support reaches the bounded one-run evidence path,
- expect one more explicit operator action string instead of repeating the evidence.

**Exit criterion:**
One focused failing controller test for the recent one-run wording/mapping gap.

---

### Task 3: GREEN minimal one-run action clarification

**Status:** completed

**Objective:** Land the smallest wording or mapping change that makes the recent single-run path more actionable.

**Files:**
- Modify: `gui/workspace_history_controller.py`
- Verify: `tests/test_workspace_history_controller.py`

**Implementation target:**
- stay inside existing evidence/next-action helpers,
- keep behavior bounded to the recent single-run path,
- do not change stale single-run behavior, compare-readiness branches, or freshness wording.

**Exit criterion:**
Focused test passes and existing workspace-history decision contracts remain green.

---

### Task 4: Verify non-regression and sync pointer-chain

**Status:** completed

**Objective:** Verify the seam and keep next-cycle docs unambiguous after the bounded slice lands.

**Files:**
- Verify: `tests/test_workspace_history_controller.py`
- Modify: this file
- Modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Verification pack:**
```bash
pytest tests/test_workspace_history_controller.py -q
ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py
```

**Exit criterion:**
Recent one-run action seam is green and docs identify the most recently delivered bounded slice correctly.

---

## Closure / successor rule

After this delivery:
- do not reopen the shipped recent one-run wording seam itself unless a focused regression appears,
- do not expand this slice into a generic history UX rewrite,
- choose another trust-surface successor only if it stays on one existing evidence/decision seam.

---

## Success definition

This slice is successful if:
- the recent one-run path becomes action-oriented instead of repeating evidence,
- stale single-run and compare-readiness contracts remain intact,
- workspace-history tests stay green,
- docs clearly show this slice as delivered rather than leaving it implied as active forever.
