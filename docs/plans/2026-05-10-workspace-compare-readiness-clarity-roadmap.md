# PromptManager Workspace Compare-Readiness Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce hesitation on compare decisions by making one bounded missing-baseline evidence path more explicit on the existing workspace history decision-support seam, without reopening prompt-detail handoff work, controller-wide policy, or broader history UX.

**Architecture:** This roadmap stays on the existing trust-surface seam in `gui/workspace_history_controller.py`, especially where decision summaries, provenance cues, run summaries, and compare-oriented next actions are derived from recent execution history. The slice must stay bounded: audit the current compare-readiness contract, pick one missing-baseline gap, reproduce it with a focused test, and land one compact wording or mapping fix without restructuring history loading or prompt-detail rendering.

**Tech Stack:** Python 3.13, PySide6 workspace/history controller, pytest, Ruff, PromptManager docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-detail-fork-handoff-clarity-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/product-ssot.md` keeps trust surfaces and bounded decision support inside the prompt-asset loop.
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` says the next slice should strengthen operator confidence more than feature breadth.
- `gui/workspace_history_controller.py` already emits compact evidence and next-action strings for compare readiness, including:
  - `Evidence: only one run available`
  - `Evidence: no comparable baseline yet`
  - `Evidence: missing rating for comparison`
  - `Evidence: missing duration for comparison`
  - `Run a different prompt version before comparing`
  - `Add ratings before comparing`
  - `Run again before comparing`
- `tests/test_workspace_history_controller.py` already covers multiple evidence and freshness contracts, so the next safe move is to extend that existing test surface rather than creating a new product seam.

So the next useful move is:

**workspace trust surface -> make one missing-baseline compare path more explicit without broadening history UX**

---

## Scope guardrails

This roadmap may improve only:
- compare-readiness wording/mapping inside `gui/workspace_history_controller.py`,
- focused controller tests in `tests/test_workspace_history_controller.py`,
- docs pointers for the bounded slice after it lands.

This roadmap must not introduce:
- new widgets,
- prompt-detail handoff rewrites,
- history model/persistence changes,
- broad controller refactors,
- new compare workflows beyond the existing cues.

---

## Main product question for this roadmap

> what is the smallest compare-readiness clarification that makes the missing-baseline path more actionable without expanding the workspace-history model?

Preferred answer shape:
- one controller file,
- one focused test file,
- one compact evidence/next-action clarification,
- no new UI surface.

---

## Stage A — Missing-baseline compare clarity

### Task 1: Audit the current missing-baseline contract

**Status:** completed

**Objective:** Confirm the exact current evidence and next-action strings for the missing-baseline path and identify the smallest wording gap.

**Files:**
- Inspect: `gui/workspace_history_controller.py`
- Inspect: `tests/test_workspace_history_controller.py`
- Maintain: `docs/plans/2026-05-10-workspace-compare-readiness-clarity-roadmap.md`

**Audit checks:**
- confirm where `Evidence: no comparable baseline yet` is emitted,
- confirm what next action currently maps from that evidence,
- confirm whether the current wording is ambiguous about what the operator should produce next.

**Exit criterion:**
One exact current string contract and one bounded clarification target.

---

### Task 2: Reproduce the gap with a focused RED controller test

**Status:** completed

**Objective:** Add one failing test that proves the current missing-baseline compare guidance is less explicit than desired.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`
- Verify: `gui/workspace_history_controller.py`

**RED target shape:**
- construct the existing missing-baseline history scenario,
- assert current decision support reaches the bounded evidence path,
- expect one more explicit baseline-oriented operator action string.

**Exit criterion:**
One focused failing controller test for the missing-baseline wording/mapping gap.

---

### Task 3: GREEN minimal compare-readiness clarification

**Status:** completed

**Objective:** Land the smallest wording or mapping change that makes the missing-baseline path more actionable.

**Files:**
- Modify: `gui/workspace_history_controller.py`
- Verify: `tests/test_workspace_history_controller.py`

**Implementation target:**
- stay inside existing evidence/next-action helpers,
- keep behavior bounded to the missing-baseline path,
- do not change unrelated rating/duration/freshness branches.

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
Missing-baseline compare-readiness seam is green and docs identify the most recently delivered bounded slice correctly.

---

## Closure / successor rule

After this delivery:
- do not reopen the shipped missing-baseline wording seam itself unless a focused regression appears,
- do not expand this slice into a generic history UX rewrite,
- choose another trust-surface successor only if it stays on one existing evidence/decision seam.

---

## Success definition

This slice is successful if:
- one missing-baseline compare path becomes more explicit,
- existing rating/duration/freshness contracts remain intact,
- workspace-history tests stay green,
- docs clearly show this slice as delivered rather than leaving it implied as active forever.
