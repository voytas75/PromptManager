# PromptManager Workspace-History Controller Typing Cleanup Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce mechanical strict-typing risk on the active inspect/run-summary seam by cleaning up `gui/workspace_history_controller.py` with small typed metadata helpers and without reopening broader prompt-chain or history-model refactors.

**Architecture:** This roadmap stays on one bounded inspect/history seam. It targets the pyright-reported unknown and partially unknown types concentrated around execution metadata extraction inside `gui/workspace_history_controller.py`, especially `metadata -> context -> execution/run` traversal for run summaries and prompt-version extraction. The slice should stay mechanical: add narrow typed coercion helpers, keep freshness/decision semantics aligned with existing tests, and avoid expanding into unrelated lineage or prompt-chain debt.

**Tech Stack:** Python 3.13, PySide6 detail/history seam, pyright, pytest, Ruff, PromptManager product docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-prompt-actions-controller-typing-cleanup-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` keeps active-seam typing cleanup subordinate to the prompt-asset product loop.
- A focused pyright audit after C3.2 showed `gui/workspace_history_controller.py` as the next strongest bounded candidate on the active inspect/run-support seam.
- The bounded pyright pack was concentrated in repeated metadata traversal patterns:
  - `metadata`
  - `context`
  - `execution`
  - `run`
  - `prompt_version`
  - `conversation_messages`
- `tests/test_workspace_history_controller.py` already covers freshness cues, run-summary ordering, limited-evidence guidance, and selection clearing, so this seam can be verified without opening new UI surface work.
- `gui/dialogs/prompt_chains.py` still carries a larger and riskier mixed debt cluster and should stay out of scope for this slice.

So the next useful move is:

**active-seam typing cleanup -> workspace history controller metadata extraction first**

---

## Scope guardrails

This roadmap may improve only:
- unknown / partially unknown metadata traversal in `gui/workspace_history_controller.py`,
- the smallest helper surface needed to coerce execution metadata into typed maps/scalars,
- focused tests / pyright verification for the workspace-history seam.

This roadmap must not introduce:
- new inspect behavior,
- prompt-chain refactors,
- lineage-summary rewrites,
- broad cleanup of unrelated history-controller sections,
- behavior changes beyond restoring already-tested freshness and next-action contracts.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- prompt-list delegate/model typing cleanup is already shipped,
- prompt-actions controller typing cleanup is already shipped,
- workspace handoff stale-validation wording is already shipped,
- prompt-detail and prompt-chain slices remain delivered previous work and must not be reopened here.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> what is the smallest typing-cleanup slice that reduces mechanical risk on the inspect/run-summary seam without changing operator-facing decision support?

Preferred answer shape:
- one bounded file,
- one repeated metadata-extraction cluster,
- tiny typed helpers instead of refactors,
- nearby tests stay green.

---

## Stage A — Workspace-history controller typing cleanup

### Task 1: Audit the workspace-history typing seam

**Status:** completed

**Objective:** Confirm the smallest bounded pyright debt cluster in `gui/workspace_history_controller.py` and identify the highest-value repeated extraction seam.

**Files:**
- Inspect: `gui/workspace_history_controller.py`
- Inspect: `tests/test_workspace_history_controller.py`
- Maintain: `docs/plans/2026-05-10-workspace-history-controller-typing-cleanup-roadmap.md`

**Implemented:**
- confirmed the bounded target is `gui/workspace_history_controller.py`,
- classified the smallest repeated debt cluster as execution metadata extraction inside `_build_run_summary(...)` and `_extract_prompt_version(...)`,
- confirmed `tests/test_workspace_history_controller.py` as the natural verification surface.

**Exit criterion:**
One bounded implementation plan for the workspace-history seam with no broader controller refactor.

---

### Task 2: Reproduce the bounded pyright failure pack

**Status:** completed

**Objective:** Lock the typing debt with a focused pyright command before code changes.

**Files:**
- Verify: `gui/workspace_history_controller.py`

**Verification command:**
```bash
.venv/bin/pyright gui/workspace_history_controller.py
```

**Observed RED baseline:**
- reproduced a bounded 23-error pyright pack,
- failures concentrated in `metadata/context/execution/run` extraction and `list[Unknown]` return typing,
- the debt pattern was repeated rather than spread across unrelated seams.

**Exit criterion:**
A reproducible, bounded pyright failure set tied to the workspace-history seam.

---

### Task 3: GREEN minimal typing cleanup

**Status:** completed

**Objective:** Remove the bounded pyright errors from the workspace-history seam without changing inspect/run-support behavior.

**Files:**
- Modify: `gui/workspace_history_controller.py`
- Verify: `tests/test_workspace_history_controller.py`

**Implemented:**
- added narrow metadata coercion helpers for child maps, ints, and text extraction,
- rewired `_build_run_summary(...)` to use typed metadata helpers instead of raw `dict[Unknown, Unknown]` chaining,
- rewired `_extract_prompt_version(...)` to the same bounded helper path,
- narrowed the execution-history list return to `list[object]`,
- restored the existing tested freshness contract:
  - naive timestamps produce no freshness cue,
  - recent hourly runs stay `recent`,
  - older multi-day single-run evidence still surfaces `stale` and preserves validation-first next actions.

**Outcome:**
- `.venv/bin/pyright gui/workspace_history_controller.py` -> `0 errors, 0 warnings, 0 informations`
- inspect/run-support behavior remained aligned with the existing test contract

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
.venv/bin/pyright gui/workspace_history_controller.py
pytest tests/test_workspace_history_controller.py -q
ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py
```

**Verified:**
- workspace-history bounded pyright pack is green,
- workspace-history tests stay green (`24 passed`),
- Ruff stays green,
- docs now identify this ledger as the most recently delivered bounded typing-cleanup slice.

---

## Current recommended next slice

This bounded typing-cleanup seam is delivered.

Do not treat this ledger as an open implementation pointer anymore.
The next bounded slice must be selected in a later revision of the direction note and/or a fresh execution ledger.

---

## Closure / successor rule

After this delivery:
- do not reopen the shipped workspace-history metadata typing seam itself unless a focused regression appears,
- do not treat this slice as permission to expand into prompt-chain or broad history-controller cleanup automatically,
- choose the next typing-cleanup slice only if it stays bounded to one active-seam file or one very small supporting contract,
- prefer the next smallest active seam over broad repo-wide cleanup.

---

## Success definition

This slice is successful if:
- `gui/workspace_history_controller.py` loses the chosen bounded pyright debt cluster,
- inspect/run-summary behavior stays aligned with current tests,
- nearby workspace-history tests still pass,
- pointer-chain docs clearly show this slice as delivered and not active.
