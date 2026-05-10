# PromptManager Prompt-Actions Controller Typing Cleanup Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce mechanical strict-typing risk on the active execution/reuse seam by cleaning up `gui/prompt_actions_controller.py` with only the smallest necessary boundary helper extraction in nearby history code.

**Architecture:** This roadmap stays on one bounded execution/reuse seam. It targets the two pyright-reported issues in `gui/prompt_actions_controller.py`: a Qt action-selection flow that does not match current stub typing and a private helper boundary crossing for validation-freshness messaging. The slice should stay mechanical: align action-selection control flow with the typed Qt surface, expose a minimal module-level freshness helper for cross-controller reuse, and preserve existing workspace-handoff behavior and copy/execute semantics.

**Tech Stack:** Python 3.13, PySide6 action/menu seam, pyright, pytest, Ruff, PromptManager product docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-prompt-list-delegate-typing-cleanup-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` keeps active-seam typing cleanup as supporting work only when it stays bounded to one active file plus the smallest necessary supporting contract.
- A focused pyright audit after C3.1 showed `gui/prompt_actions_controller.py` as the next smallest active-seam typing cluster.
- The bounded failure pack contains only two errors:
  - an unnecessary `None` check after `QMenu.exec(...)`,
  - private helper usage across the controller/history boundary.
- `tests/test_prompt_actions_controller.py` already covers the workspace-handoff freshness cue and menu/action behavior, so this seam can be verified without opening broader GUI churn.
- `gui/workspace_history_controller.py` still has larger unrelated typing debt; this slice must not expand into a history-controller cleanup.

So the next useful move is:

**active-seam typing cleanup -> prompt actions controller first**

---

## Scope guardrails

This roadmap may improve only:
- missing or mismatched type/control-flow assumptions in `gui/prompt_actions_controller.py`,
- the smallest supporting helper seam in `gui/workspace_history_controller.py` needed to avoid private cross-controller access,
- focused tests / pyright verification for the prompt-actions seam.

This roadmap must not introduce:
- action-menu behavior changes,
- workspace-handoff wording changes beyond preserving current stale/generic routing,
- broad typing churn in `gui/workspace_history_controller.py`,
- new execution flows,
- refactors whose main purpose is style rather than type-risk reduction.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- prompt copying stays locked to the stored prompt body,
- workspace handoff uses the stronger stale-validation cue when recent evidence is stale,
- prompt-list delegate/model typing cleanup is already shipped,
- prompt-detail and prompt-chain slices remain delivered previous work and must not be reopened here.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> what is the smallest typing-cleanup slice that reduces mechanical risk on an active execute/reuse seam without changing action behavior or reopening history-controller debt?

Preferred answer shape:
- one bounded file first,
- one tiny pyright debt cluster,
- one minimal supporting boundary helper if unavoidable,
- nearby tests stay green.

---

## Stage A — Prompt-actions controller typing cleanup

### Task 1: Audit the prompt-actions typing seam

**Status:** completed

**Objective:** Confirm the smallest pyright debt cluster in `gui/prompt_actions_controller.py` and classify whether any nearby support file must move with it.

**Files:**
- Inspect: `gui/prompt_actions_controller.py`
- Inspect: `gui/workspace_history_controller.py`
- Inspect: `tests/test_prompt_actions_controller.py`
- Maintain: `docs/plans/2026-05-10-prompt-actions-controller-typing-cleanup-roadmap.md`

**Implemented:**
- confirmed the bounded cluster is only two pyright errors,
- classified them as one Qt action-selection control-flow mismatch and one private helper boundary violation,
- confirmed the natural verification surface is `tests/test_prompt_actions_controller.py`.

**Exit criterion:**
One bounded implementation plan for the prompt-actions seam with a call on whether a tiny helper extraction is required.

---

### Task 2: Reproduce the bounded pyright failure pack

**Status:** completed

**Objective:** Lock the typing debt with a focused pyright command before code changes.

**Files:**
- Verify: `gui/prompt_actions_controller.py`

**Verification command:**
```bash
.venv/bin/pyright gui/prompt_actions_controller.py
```

**Observed RED baseline:**
- reproduced a bounded 2-error pyright pack,
- line 140 reported an unnecessary `None` comparison on `selected_action`,
- line 281 reported `reportPrivateUsage` for `_build_validation_freshness_summary`.

**Exit criterion:**
A reproducible, bounded pyright failure set tied to the prompt-actions seam.

---

### Task 3: GREEN minimal typing cleanup

**Status:** completed

**Objective:** Remove the bounded pyright errors from the prompt-actions seam without changing action behavior.

**Files:**
- Modify: `gui/prompt_actions_controller.py`
- Modify: `gui/workspace_history_controller.py`
- Verify: `tests/test_prompt_actions_controller.py`

**Implemented:**
- aligned the `QMenu.exec(...)` action-selection flow with the typed Qt surface by removing the unnecessary `None` guard and comparing selected actions directly,
- replaced the cross-controller private helper call with a small module-level `build_validation_freshness_summary(...)` helper,
- kept a backward-compatible wrapper in `WorkspaceHistoryController` so existing history-controller behavior stays unchanged,
- avoided any wider cleanup of unrelated unknown/dict-shaped history metadata debt.

**Outcome:**
- `.venv/bin/pyright gui/prompt_actions_controller.py` -> `0 errors, 0 warnings, 0 informations`
- prompt-actions behavior and workspace-handoff messaging stayed unchanged

---

### Task 4: Verify non-regression and sync pointer-chain

**Status:** completed

**Objective:** Verify the seam and keep next-cycle docs unambiguous after the bounded slice lands.

**Files:**
- Verify: `tests/test_prompt_actions_controller.py`
- Modify: this file
- Modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Verification pack:**
```bash
.venv/bin/pyright gui/prompt_actions_controller.py
pytest tests/test_prompt_actions_controller.py -q
ruff check gui/prompt_actions_controller.py gui/workspace_history_controller.py tests/test_prompt_actions_controller.py
```

**Verified:**
- prompt-actions bounded pyright pack is green,
- prompt-actions tests stay green (`12 passed`),
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
- do not reopen the shipped prompt-actions typing seam itself unless a focused regression appears,
- do not treat the helper extraction in `gui/workspace_history_controller.py` as permission to expand into broader history-controller typing cleanup inside this slice,
- choose the next typing-cleanup slice only if it stays bounded to one active-seam file or one very small supporting contract,
- prefer the next smallest active seam over broad repo-wide cleanup.

---

## Success definition

This slice is successful if:
- `gui/prompt_actions_controller.py` loses the chosen bounded pyright debt cluster,
- action behavior and workspace-handoff wording do not change,
- nearby prompt-actions tests still pass,
- pointer-chain docs clearly show this slice as delivered and not active.
