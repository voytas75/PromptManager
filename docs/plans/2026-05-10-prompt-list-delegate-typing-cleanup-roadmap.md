# PromptManager Prompt-List Delegate Typing Cleanup Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce mechanical strict-typing risk on the active retrieval/list seam by cleaning up `gui/prompt_list_delegate.py` first, with only the smallest necessary type-supporting touch in nearby list-model code.

**Architecture:** This roadmap stays on one bounded retrieval/list seam. It targets pyright-reported unknown and partially unknown types in `gui/prompt_list_delegate.py`, especially around `QModelIndex` access, preview/handoff cue extraction, and match-span coercion. The slice should stay mechanical: clarify types, narrow `None` handling, and reuse existing list-model contracts rather than changing retrieval behavior or adding new UI features.

**Tech Stack:** Python 3.13, PySide6 model/view seam, pyright, pytest, Ruff, PromptManager product docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-prompt-chain-result-decision-clarity-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` keeps Candidate 3 pending for active-seam typing cleanup.
- Candidate 3 boundaries explicitly say:
  - choose smallest bounded files first,
  - avoid broad repo churn,
  - prefer files on retrieval/detail/execution seams.
- A focused pyright audit on active seams showed `gui/prompt_list_delegate.py` as one of the smallest high-value files with concentrated strict-typing debt.
- The same audit showed larger debt in `gui/dialogs/prompt_chains.py`, but that file is broader and riskier for a first C3 slice.
- `gui/prompt_list_model.py` and the nearby list-side tests already exist, so the best next move is a bounded delegate-first typing cleanup rather than a bigger refactor.

So the next useful move is:

**active-seam typing cleanup -> retrieval/list delegate first**

---

## Scope guardrails

This roadmap may improve only:
- missing or unknown type annotations in `gui/prompt_list_delegate.py`,
- the smallest nearby type-supporting contract in `gui/prompt_list_model.py` if the delegate cannot be typed cleanly otherwise,
- focused tests / pyright verification for the list-side seam.

This roadmap must not introduce:
- retrieval ranking changes,
- search/result behavior changes,
- new list UI surfaces,
- broad typing churn across unrelated files,
- refactors whose main purpose is style rather than type-risk reduction.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- prompt-list action clarity beyond title match is already shipped,
- prompt-list delegate already renders the bounded handoff cue row,
- prompt-list model/delegate tests already cover current retrieval cue behavior,
- prompt-detail and prompt-chain slices are delivered previous work and should not be reopened here.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> what is the smallest typing-cleanup slice that reduces mechanical risk on an active retrieval/list seam without changing retrieval behavior or opening a larger refactor?

Preferred answer shape:
- one bounded file first,
- one small pyright debt cluster,
- no behavior drift,
- nearby tests stay green.

---

## Stage A — Prompt-list delegate typing cleanup

### Task 1: Audit the prompt-list delegate typing seam

**Status:** completed

**Objective:** Confirm the smallest pyright debt cluster in `gui/prompt_list_delegate.py` and decide whether any nearby model contract must be typed alongside it.

**Files:**
- Inspect: `gui/prompt_list_delegate.py`
- Inspect: `gui/prompt_list_model.py`
- Inspect: `tests/test_prompt_list_model.py`
- Maintain: `docs/plans/2026-05-10-prompt-list-delegate-typing-cleanup-roadmap.md`

**Implemented:**
- confirmed the smallest bounded debt cluster sits in `gui/prompt_list_delegate.py`,
- classified the errors into four groups: missing index annotations, unknown `index.data(...)` values, `str | None` flow into string-only helpers, and tuple/span coercion ambiguity,
- confirmed `gui/prompt_list_model.py` only needs a minimal supporting parent-type annotation.

**Exit criterion:**
One bounded implementation plan for the delegate seam, plus a call on whether `gui/prompt_list_model.py` needs a minimal supporting type adjustment.

---

### Task 2: Reproduce the bounded pyright failure pack

**Status:** completed

**Objective:** Lock the typing debt with a focused pyright command before code changes.

**Files:**
- Verify: `gui/prompt_list_delegate.py`
- Maybe verify: `gui/prompt_list_model.py`

**Verification command:**
```bash
.venv/bin/pyright gui/prompt_list_delegate.py gui/prompt_list_model.py
```

**Observed RED baseline:**
- reproduced a bounded 33-error pyright pack,
- almost all failures were concentrated in `gui/prompt_list_delegate.py`,
- `gui/prompt_list_model.py` contributed only the small `parent` annotation issue.

**Exit criterion:**
A reproducible, bounded pyright failure set tied to the delegate seam.

---

### Task 3: GREEN minimal typing cleanup

**Status:** completed

**Objective:** Remove the bounded pyright errors from the delegate seam without changing list behavior.

**Files:**
- Modify: `gui/prompt_list_delegate.py`
- Modify: `gui/prompt_list_model.py`
- No test contract changes were required in `tests/test_prompt_list_model.py`

**Implemented:**
- added explicit index-like typing for the public helper and Qt override signatures,
- introduced local typed text coercion for `index.data(...)` reads used by preview and handoff cue rendering,
- narrowed the `str | None` flow before calling `_elide_text_and_spans(...)`,
- simplified `_coerce_match_spans(...)` so pyright can validate the tuple shape without behavior changes,
- added `parent: QObject | None` typing in `PromptListModel.__init__`.

**Outcome:**
- `.venv/bin/pyright gui/prompt_list_delegate.py gui/prompt_list_model.py` -> `0 errors, 0 warnings, 0 informations`
- retrieval/list behavior stayed unchanged

---

### Task 4: Verify non-regression and sync pointer-chain

**Status:** completed

**Objective:** Verify the seam and keep next-cycle docs unambiguous after the bounded slice lands.

**Files:**
- Verify: `tests/test_prompt_list_model.py`
- Modify: this file
- Modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Verification pack:**
```bash
.venv/bin/pyright gui/prompt_list_delegate.py gui/prompt_list_model.py
pytest tests/test_prompt_list_model.py -q
ruff check gui/prompt_list_delegate.py gui/prompt_list_model.py tests/test_prompt_list_model.py
```

**Verified:**
- pyright bounded pack is green,
- list-side tests stay green (`21 passed`),
- Ruff stays green,
- docs now point to one active bounded typing-cleanup ledger during implementation.

---

## Current recommended next slice

This bounded typing-cleanup seam is delivered.

Do not treat this ledger as an open implementation pointer anymore.
The next bounded slice must be selected in a later revision of the direction note and/or a fresh execution ledger.

---

## Closure / successor rule

After this delivery:
- do not reopen the shipped prompt-list delegate typing seam itself unless a focused regression appears,
- choose the next typing-cleanup slice only if it stays bounded to one active-seam file or one very small supporting contract,
- prefer the next smallest active seam (for example another retrieval/detail/execution file) over broad repo-wide cleanup.

---

## Success definition

This slice is successful if:
- `gui/prompt_list_delegate.py` loses the chosen bounded pyright debt cluster,
- retrieval/list behavior does not change,
- nearby list tests still pass,
- pointer-chain docs clearly show one active bounded typing-cleanup ledger.
