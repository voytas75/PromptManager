# PromptManager Search-Result Action Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Strengthen operator confidence directly in the prompt list when search results match by source or scenario, so the user can better tell whether to open detail first or expect likely immediate reuse, without changing retrieval ranking, persistence, or the broader product model.

**Architecture:** This roadmap stays inside the existing prompt-list retrieval seam. It should improve action clarity by extending the existing list-local match-reason/handoff pattern rather than duplicating detail-view decision logic. Each slice should remain small, testable, and local to the operator flow: find -> judge likely next move -> open detail or reuse.

**Tech Stack:** Python 3.13, PySide6 GUI seams, PromptManager retrieval/detail controllers, pytest, Ruff, existing product SSOT docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Previous completed bounded ledger: `docs/plans/2026-05-10-search-detail-reuse-confidence-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/product-ssot.md` keeps retrieval/discovery, inspect clarity, and reuse/refine inside Priority 1.
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` now points at search-result action clarity beyond title match as the current recommended bounded slice.
- the prior bounded ledger under `docs/plans/2026-05-10-search-detail-reuse-confidence-roadmap.md` is closed for that delivered slice.
- the current prompt-list seam already exposes one bounded handoff cue for `Matched in title`, but does not yet provide equivalent action confidence for `Matched in source` or `Matched in scenario`.

So the next useful move is:

**asset-first core -> prompt-list action clarity for non-title matches on existing seams**

---

## Scope guardrails

This roadmap may improve only:
- prompt-list handoff clarity for search-result matches,
- one bounded operator cue for non-title matches,
- compact wording or rendering improvements that stay list-local,
- small parity guards proving the cue remains GUI-local.

This roadmap must not introduce:
- ranking changes,
- a new retrieval backend or explainability layer,
- new persistence,
- a retrieval dashboard,
- reuse of full detail-view `Decision` or `Recommended next action` strings,
- workflow-engine semantics,
- new automation surfaces as the primary output of this cycle.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- prompt-list search results already expose compact `Matched in source` / `Matched in scenario` / `Matched in title` reasons,
- title matches already expose one list-local handoff cue: `Likely reusable as-is`,
- detail view already exposes bounded `Decision`, provenance, and next-action cues,
- `Copy Prompt` and `Open in Workspace` already exist as immediate reuse actions,
- retrieval/detail seams already have focused test coverage,
- the previous `search/detail/reuse confidence` slice is landed and closed.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> when a search result matches by source or scenario rather than title, how do we reduce hesitation between “this may be relevant” and “I know whether I should inspect first or expect immediate reuse” without adding a bigger retrieval system?

Preferred answer shape:
- one bounded non-title-match cue family,
- one clearer list-side action expectation,
- no duplication of detail-view semantics,
- all on the current prompt-list surface.

---

## Stage A — Non-title search-result action clarity

### Task 1: Audit the remaining non-title-match hesitation seam

**Status:** completed

**Objective:** Confirm where `Matched in source` and `Matched in scenario` still leave operator hesitation after the landed title-match cue slice.

**Chosen bounded hesitation seam (confirmed):**
- The prompt list already exposes compact source-of-match reasons for all three active-search paths:
  - `Matched in source`
  - `Matched in scenario`
  - `Matched in title`
- Only the title-match path currently exposes any list-local handoff confidence:
  - `Likely reusable as-is`
- For source and scenario matches, the list still tells the operator *where* relevance came from, but not whether the likely next move is immediate reuse or inspect-first review.
- This leaves one bounded hesitation seam: a non-title result can look plausibly relevant from source/scenario context, yet the operator still has no compact list-side signal that it likely needs inspection before reuse.
- The selected v1 direction is therefore **one list-local inspect-first confidence cue for non-title matches**, without changing retrieval ranking and without duplicating detail-view `Decision` / `Recommended next action` text.

**Audit notes (confirmed from code/tests):**
- `gui/prompt_list_model.py` currently computes `MatchReasonRole` for source/scenario/title, but `HandoffCueRole` returns a cue only for `Matched in title`.
- `gui/prompt_list_model.py` does not expose any differentiated handoff cue for `Matched in source` or `Matched in scenario`.
- `tests/test_prompt_list_model.py` locks source/scenario/title reason cues, but only title-match currently has a handoff-cue assertion.
- `gui/prompt_list_delegate.py` renders preview and title emphasis only; it does not render an extra visible non-title action cue on its own.
- `gui/prompt_list_presenter.py` keeps ordinary search status calm and inspect-oriented (`Showing search results — inspect a prompt for reuse details.`), but that is search-global state, not per-row confidence.
- `tests/test_retrieval_cues_parity.py` guards prompt-list cues as GUI-local by design, so any new non-title cue should remain on the prompt-list seam unless intentionally promoted.

**Files:**
- Inspect: `gui/prompt_list_model.py`
- Inspect: `gui/prompt_list_delegate.py`
- Inspect: `gui/prompt_list_presenter.py`
- Inspect: `tests/test_prompt_list_model.py`
- Inspect: `tests/test_retrieval_cues_parity.py`
- Maybe inspect: `tests/test_prompt_list_presenter.py`
- Maintain: `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md`

**Steps:**
1. inspect the current prompt-list cue seam,
2. confirm what action confidence is already implied for title matches only,
3. identify one bounded hesitation seam for source/scenario matches,
4. record the chosen seam under this task before implementation starts.

**Verification:**
- read the inspected files,
- search existing tests for current cue wording,
- confirm the chosen seam is not already covered by shipped behavior/tests.

---

### Task 2: Add a failing focused test for one bounded non-title cue family

**Status:** completed

**Objective:** Freeze one missing list-side action-confidence improvement before UI code changes.

**Constraint:**
Pick only one cue family for v1.
Do not open several wording/UI ideas at once.

**Implemented:**
- Added focused RED coverage in `tests/test_prompt_list_model.py` for the single chosen v1 cue family:
  - `Matched in source` -> `Inspect before reuse`
  - `Matched in scenario` -> `Inspect before reuse`
- Kept title-match coverage unchanged so the test seam still distinguishes:
  - title-match immediate-reuse confidence,
  - non-title inspect-first confidence.
- Confirmed the missing non-title cue as a real failing seam before implementation.

**Verified:**
- `pytest tests/test_prompt_list_model.py -q` -> RED before implementation (`2 failed` on missing non-title handoff cue).

**Files:**
- Modify: one focused test file among:
  - `tests/test_prompt_list_model.py`
  - `tests/test_prompt_list_presenter.py`
  - `tests/test_retrieval_cues_parity.py`
- Reference: the matching GUI seam chosen in Task 1

**Verification:**
Run only the focused test(s) for the chosen seam and confirm RED before implementation.

---

### Task 3: Implement the bounded non-title action cue on the existing seam

**Status:** completed

**Objective:** Add the smallest operator-facing cue that reduces hesitation for source/scenario matches without expanding the retrieval model.

**Implemented:**
- Extended `gui/prompt_list_model.py` on the existing `HandoffCueRole` seam.
- Preserved current title behavior:
  - `Matched in title` -> `Likely reusable as-is`
- Added one bounded non-title cue family:
  - `Matched in source` -> `Inspect before reuse`
  - `Matched in scenario` -> `Inspect before reuse`
- Kept the change list-local and model-local; no ranking, persistence, presenter-state, or detail-view semantics changed.

**Verified:**
- `pytest tests/test_prompt_list_model.py tests/test_retrieval_cues_parity.py -q` -> `22 passed`
- `ruff check gui/prompt_list_model.py tests/test_prompt_list_model.py tests/test_retrieval_cues_parity.py` -> OK

**Files:**
- Modify: the single GUI seam chosen in Task 1
- Modify: the focused test file from Task 2
- Maybe modify: `tests/test_retrieval_cues_parity.py` if the cue must remain GUI-local

**Implementation targets:**
- keep the cue compact,
- prefer existing wording patterns over new terminology,
- avoid duplicating search state and detail state in two places,
- preserve current retrieval ranking and persistence behavior.

**Verification:**
Run the focused tests for the changed seam, plus one nearby smoke pack.

---

### Task 4: Sync the execution ledger after the bounded slice lands

**Status:** completed

**Objective:** Keep one unambiguous pointer chain after implementation.

**Implemented:**
- Updated this ledger so the Task 2/3 slice is recorded as shipped with explicit `Implemented:` and `Verified:` notes.
- Verified the active pointer chain still resolves cleanly:
  - strategic direction: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - active bounded execution ledger: `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md`
  - umbrella pointer: `docs/plans/2026-04-25-roadmap-implementation-plan.md`
- Confirmed no README or `docs/product-ssot.md` update is needed for this slice because product truth did not change; only one bounded list-local cue family shipped.
- Re-reviewed the strategic direction note and confirmed it still points at the same active bounded ledger, so no direction-file rewrite was required.

**Verified:**
- searched `docs/plans/` for stale active-slice wording and found no conflicting active execution ledger for this slice.
- re-read:
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`
  - `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md`
- confirmed the remaining strategic note still marks `Search-result action clarity beyond title match` as the current recommended bounded slice, which is consistent with this ledger staying active until the next bounded successor is explicitly chosen.

**Files:**
- Modify: `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md`
- Maybe modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Maybe modify: `README.md` only if user-visible positioning changed
- Maybe modify: `docs/product-ssot.md` only if product truth changed

**Steps:**
1. update the implemented task statuses,
2. add `Implemented:` and `Verified:` notes,
3. keep README/SSOT changes minimal unless product truth changed,
4. verify no older note still points at an already-closed “next slice”.

**Verification:**
- search `docs/plans/` for stale next-slice phrases,
- confirm this file remains the only active execution ledger for this slice,
- confirm there is no docs split-brain.

---

### Task 5: Add a failing focused test for visible list-row handoff cue legibility

**Status:** completed

**Objective:** Freeze one bounded visibility seam so the existing non-title handoff cue becomes actually legible on the prompt-list row.

**Constraint:**
- Keep the slice on the existing prompt-list delegate seam.
- Do not introduce a new cue family or change ranking/model semantics.
- Prefer one subtle visible cue path over broader row redesign.

**Implemented:**
- Added focused delegate-facing RED coverage in `tests/test_prompt_list_model.py` for one bounded visibility seam.
- Locked that a non-title search result with existing model cue `Inspect before reuse` must expose the same cue through a delegate-visible helper.
- Kept the test seam narrow: it verifies row-level visibility plumbing without opening broader rendering/layout changes yet.

**Verified:**
- `pytest tests/test_prompt_list_model.py -q` -> RED before implementation (`1 failed` on missing `PromptListDelegate.handoff_cue_text`).

**Files:**
- Modify: `tests/test_prompt_list_model.py`
- Modify: `gui/prompt_list_delegate.py`
- Reference: `gui/prompt_list_model.py`

**Verification:**
Run only the focused delegate/model tests for the chosen visibility seam and confirm RED before implementation.

---

### Task 6: Render the bounded handoff cue on the existing prompt-list row seam

**Status:** completed

**Objective:** Make the existing list-local handoff cue visibly legible in the row without expanding retrieval/detail semantics.

**Implemented:**
- Extended `gui/prompt_list_delegate.py` with a bounded public helper `handoff_cue_text(index)` that exposes the row-visible cue text from the existing `HandoffCueRole` seam.
- Added a subtle third line in delegate rendering for the handoff cue when present.
- Updated `sizeHint()` so rows reserve extra height when the visible handoff cue line is present.
- Preserved current retrieval/model semantics and kept the change delegate-local.

**Verified:**
- `pytest tests/test_prompt_list_model.py -q` -> `21 passed`
- `ruff check gui/prompt_list_delegate.py tests/test_prompt_list_model.py` -> OK

**Implementation targets:**
- keep the cue subtle and compact,
- prefer reusing the current title/preview row structure,
- preserve current ranking, persistence, and presenter behavior,
- avoid broad list-row redesign.

**Files:**
- Modify: `gui/prompt_list_delegate.py`
- Modify: `tests/test_prompt_list_model.py`
- Maybe modify: nearby presenter/delegate smoke tests only if needed

**Verification:**
Run the focused delegate/model tests for the changed seam, plus one nearby smoke pack.

---

### Task 7: Add actual row-height/render contract coverage for the visible handoff cue

**Status:** completed

**Objective:** Prove the visible handoff cue is not only exposed by the delegate helper, but also affects real row layout on the prompt-list seam.

**Implemented:**
- Added focused test coverage in `tests/test_prompt_list_model.py` proving that a row with visible handoff cue reserves more height than a comparable row without that cue.
- Kept the contract bounded to layout/visibility rather than pixel-level painting assertions.

**Verified:**
- `pytest tests/test_prompt_list_model.py -q` -> RED before implementation (`1 failed` because cue rows did not reserve extra height).
- `pytest tests/test_prompt_list_model.py -q` -> GREEN after implementation (`21 passed`).
- `ruff check gui/prompt_list_delegate.py tests/test_prompt_list_model.py` -> OK

---

## Current recommended next slice

**Status:** delivered for the current v1 seam

The bounded `non-title search-result action cue v1` slice is now landed on the active prompt-list seam:
- non-title handoff confidence exists in `gui/prompt_list_model.py`,
- the cue is visible on the prompt-list row through `gui/prompt_list_delegate.py`,
- focused and nearby smoke tests pass.

**Next successor to choose explicitly:**
- either one small follow-up on the same prompt-list/detail confidence seam,
- or move to `Detail refine/fork action clarity` if that is the better bounded next product question.

Selection rule for the next successor:
- do not reopen the shipped v1 cue family itself,
- prefer the smallest remaining hesitation seam,
- avoid any slice that would require ranking changes, new persistence, or parallel retrieval logic.

---

## Update workflow after each implemented slice

After each completed task or small task bundle:
1. update this roadmap (`Status`, `Implemented`, `Verified`),
2. update `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` only if the chosen active slice pointer changes,
3. update `README.md` only if user-visible product emphasis changed,
4. update `docs/product-ssot.md` only if product truth changed,
5. verify no older roadmap still advertises a conflicting active next slice.

---

## Definition of done

This roadmap slice is done only when:
- one bounded non-title action-confidence improvement has shipped on an existing prompt-list seam,
- focused tests pass,
- nearby smoke tests pass,
- Ruff passes on touched files,
- the docs pointer chain remains unambiguous,
- no broader retrieval/workflow/product drift was introduced.
