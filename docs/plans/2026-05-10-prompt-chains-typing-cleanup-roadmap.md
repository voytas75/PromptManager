# PromptManager Prompt-Chains Typing Cleanup Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce mechanical strict-typing risk in `gui/dialogs/prompt_chains.py` with bounded micro-passes that preserve prompt-chain dialog behavior, keep supporting-summary UX intact, and avoid broad dialog refactors.

**Architecture:** This roadmap stays on one bounded prompt-chain dialog seam. It targets pyright-reported nullability mismatches, partially unknown callable/default patterns, recent-run record coercion, payload import typing, and reasoning-payload traversal inside `gui/dialogs/prompt_chains.py`. The slice should stay mechanical: remove unnecessary Qt nullability branches, add small typed helpers for status messages and payload coercion, and keep all existing prompt-chain semantics, especially `Supporting summary` vs `Final output`, recent-run rendering, clipboard actions, and reasoning-summary extraction.

**Tech Stack:** Python 3.13, PySide6 dialog/widgets, pyright, pytest, Ruff, PromptManager product docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-workspace-history-controller-typing-cleanup-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` keeps prompt-chain work as a bounded enabler rather than the main product story.
- A focused pyright audit after C3.3 showed the remaining active typing debt concentrated in `gui/dialogs/prompt_chains.py`.
- The bounded failure pack initially contained 51 errors, but the structure was naturally separable into micro-clusters:
  - unnecessary Qt nullability comparisons,
  - partially unknown `llm_status_message` fallback callables,
  - recent-run history record coercion,
  - JSON payload import typing,
  - reasoning-payload traversal.
- `tests/test_prompt_chain_dialog.py` already covers dialog behavior across clipboard actions, recent-run rendering, and supporting-summary reasoning display, so this seam can be verified without reopening unrelated GUI surfaces.

So the next useful move is:

**prompt-chain dialog typing cleanup -> bounded micro-passes on one file**

---

## Scope guardrails

This roadmap may improve only:
- strict-typing / partially unknown debt in `gui/dialogs/prompt_chains.py`,
- tiny typed helpers required to keep the dialog seam mechanical and bounded,
- focused prompt-chain dialog verification.

This roadmap must not introduce:
- changes to prompt-chain product semantics,
- changes to `Supporting summary` / `Final output` intent,
- new prompt-chain workflows,
- broad refactors of the dialog architecture,
- cross-file cleanup outside the smallest supporting contract already inside this dialog file.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- `Supporting summary` remains secondary to `Final output`,
- prompt-list, prompt-actions, and workspace-history typing slices are already shipped,
- prompt-chain dialog tests already cover clipboard actions, reasoning summaries, and recent-run text,
- no prompt-chain-first product shift is allowed in this slice.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> what is the smallest series of typing-cleanup micro-passes that makes `gui/dialogs/prompt_chains.py` green without changing prompt-chain dialog behavior?

Preferred answer shape:
- one file only,
- micro-passes grouped by debt cluster,
- typed helpers instead of refactors,
- prompt-chain dialog tests stay green throughout.

---

## Stage A — Prompt-chains typing cleanup

### Task 1: Audit the prompt-chains typing seam

**Status:** completed

**Objective:** Confirm the smallest bounded pyright debt clusters in `gui/dialogs/prompt_chains.py` and identify a safe execution order.

**Files:**
- Inspect: `gui/dialogs/prompt_chains.py`
- Inspect: `tests/test_prompt_chain_dialog.py`
- Maintain: `docs/plans/2026-05-10-prompt-chains-typing-cleanup-roadmap.md`

**Implemented:**
- confirmed `gui/dialogs/prompt_chains.py` as the only remaining active file in the bounded typing pack,
- classified the first safe cluster as Qt nullability plus `llm_status_message` helper typing,
- classified later clusters as recent-run history record coercion and reasoning/payload traversal,
- confirmed `tests/test_prompt_chain_dialog.py` as the natural verification surface.

**Exit criterion:**
One bounded execution order for prompt-chain typing micro-passes with no broad dialog refactor.

---

### Task 2: Reproduce the bounded pyright failure pack

**Status:** completed

**Objective:** Lock the typing debt with a focused pyright command before code changes.

**Files:**
- Verify: `gui/dialogs/prompt_chains.py`

**Verification command:**
```bash
.venv/bin/pyright gui/dialogs/prompt_chains.py
```

**Observed RED baseline:**
- reproduced a bounded 51-error pyright pack,
- errors were concentrated in nullability checks, unknown fallback callables, recent-run record coercion, payload import typing, and reasoning traversal.

**Exit criterion:**
A reproducible, bounded pyright failure set tied to the prompt-chain dialog seam.

---

### Task 3: GREEN bounded micro-passes

**Status:** completed

**Objective:** Remove the bounded pyright debt from `gui/dialogs/prompt_chains.py` without changing prompt-chain dialog behavior.

**Files:**
- Modify: `gui/dialogs/prompt_chains.py`
- Verify: `tests/test_prompt_chain_dialog.py`

**Implemented:**
- removed unnecessary Qt nullability branches for `QListWidgetItem`, `PromptManager`, `QClipboard`, and `QScrollBar`,
- added `_resolve_llm_status_message(...)` and the default fallback helper to avoid partially unknown status-message callables,
- added typed recent-run helpers:
  - `_PromptChainHistoryRecord`
  - `_history_record_text(...)`
  - `_history_record_optional_text(...)`
  - `_recent_chain_run_records(...)`
- rewired `_recent_run_history(...)` to the typed recent-run helper path,
- added `_PromptChainPayload` and `_coerce_prompt_chain_payload(...)`,
- rewired JSON import payload handling to pass typed payloads into `chain_from_payload(...)`,
- rewired reasoning-summary extraction and `_search_reasoning_payload(...)` to typed payload traversal and explicit sequence casting,
- completed the micro-pass sequence from `51 -> 38 -> 13 -> 0` pyright errors without opening a broad dialog refactor.

**Outcome:**
- `.venv/bin/pyright gui/dialogs/prompt_chains.py` -> `0 errors, 0 warnings, 0 informations`
- prompt-chain dialog behavior remained aligned with existing tests and supporting-summary intent

---

### Task 4: Verify non-regression and sync pointer-chain

**Status:** completed

**Objective:** Verify the seam and keep next-cycle docs unambiguous after the bounded slice lands.

**Files:**
- Verify: `tests/test_prompt_chain_dialog.py`
- Modify: this file
- Modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Verification pack:**
```bash
.venv/bin/pyright gui/dialogs/prompt_chains.py
pytest tests/test_prompt_chain_dialog.py -q
ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
```

**Verified:**
- prompt-chain bounded pyright pack is green,
- prompt-chain dialog tests stay green (`46 passed`),
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
- do not reopen the shipped prompt-chains typing seam itself unless a focused regression appears,
- do not treat this slice as permission for broader prompt-chain dialog refactors,
- choose the next typing-cleanup slice only if it stays bounded to one active-seam file or one very small supporting contract,
- keep prompt-chain work subordinate to the prompt-asset product center.

---

## Success definition

This slice is successful if:
- `gui/dialogs/prompt_chains.py` loses the chosen bounded pyright debt clusters,
- prompt-chain dialog behavior stays aligned with current tests,
- nearby prompt-chain tests still pass,
- pointer-chain docs clearly show this slice as delivered and not active.
