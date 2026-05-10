# PromptManager Search / Detail / Reuse Confidence Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Strengthen the operator path from search or recent results into detail, next-step judgment, and immediate reuse/refine actions without changing retrieval ranking, adding new persistence, or broadening PromptManager beyond its asset-first model.

**Architecture:** This roadmap stays inside the existing retrieval/detail seams. It should improve confidence, legibility, and decision support by reusing current prompt-list, presenter, and detail-widget surfaces rather than introducing a second retrieval model or a dashboard-first layer. Each slice should remain small, testable, and local to the existing operator flow: find -> understand -> decide -> reuse/refine.

**Tech Stack:** Python 3.13, PySide6 GUI seams, PromptManager retrieval/detail controllers, pytest, Ruff, existing product SSOT docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/product-ssot.md` keeps retrieval/discovery, inspect clarity, and reuse/refine inside Priority 1.
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` sets the next-cycle priority as retrieval / inspect / reuse confidence.
- the previous broad roadmap ledger under `docs/plans/2026-04-25-roadmap-implementation-plan.md` is historically useful but no longer the right active execution ledger for the next bounded slice.
- the repo already shipped multiple trust and inspect cues, so the next useful move is not another broad SSOT rewrite but a fresh execution ledger focused on the current highest-value seam.

So the next useful move is:

**asset-first core -> retrieval/detail/reuse confidence on existing seams**

---

## Scope guardrails

This roadmap may improve only:
- search-result confidence cues,
- detail-view decision support,
- handoff clarity from search/recent into copy/open/run/refine,
- bounded reuse-vs-refine guidance from already-available context,
- small GUI-local wording/ordering improvements that reduce hesitation.

This roadmap must not introduce:
- ranking changes,
- a new retrieval backend or explainability layer,
- new persistence,
- a retrieval dashboard,
- a second product model outside the current prompt-list/detail flow,
- workflow-engine semantics,
- new automation surfaces as the primary output of this cycle.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- prompt-list retrieval/discovery confidence work already shipped in earlier bounded cycles,
- inspect/detail view already exposes bounded decision-support cues,
- `Copy Prompt` and `Open in Workspace` already exist as immediate reuse actions,
- recent/retrieval/detail seams already have focused test coverage,
- prompt-chain and structured-run work exists but is not the center of this roadmap.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> once the operator sees a likely prompt match, how do we reduce hesitation between “this looks relevant” and “I know what to do next” without adding a bigger retrieval system?

Preferred answer shape:
- one clearer confidence cue,
- one stronger detail handoff,
- one tighter next-action seam,
- all on the current surfaces.

---

## Stage A — Search/result confidence

### Task 1: Audit the current search-to-detail hesitation seam

**Status:** completed

**Objective:** Confirm where the current operator flow still leaves ambiguity between match legibility and next-step confidence.

**Chosen bounded hesitation seam (confirmed):**
- Search results already expose one compact `Matched in source` / `Matched in scenario` / `Matched in title` cue.
- Detail view already exposes bounded `Decision` / provenance / next-action cues once the operator lands there.
- The current gap is the handoff between those two surfaces: the prompt list tells *where* the match came from, but not whether opening the item is likely to lead to immediate reuse or expected refinement.
- This leaves one small hesitation seam: a result can look relevant in the list, yet the operator still has to open detail to learn whether the likely next move is reuse-as-is versus refine/fork.
- The selected v1 direction is therefore **one search-local confidence cue that previews likely next-step confidence before opening detail**, without duplicating full detail-view decision text or changing retrieval ranking.

**Audit notes (confirmed from code/tests):**
- `gui/prompt_list_model.py` currently limits list-side reasoning to `MatchReasonRole` with three source-of-match variants only.
- `tests/test_prompt_list_model.py` currently locks only those three list-side reason strings.
- `gui/widgets/prompt_detail_widget.py` already provides `Decision`, provenance, and next-action surfaces, so the missing cue is not in detail rendering itself.
- `gui/workspace_history_controller.py` already computes bounded decision/next-action summaries, but only after selection/detail hydration.
- `tests/test_retrieval_cues_parity.py` explicitly guards prompt-list cues as GUI-local, so any new search cue should stay local to the prompt-list seam unless intentionally promoted.

**Files:**
- Inspect: `gui/prompt_list_model.py`
- Inspect: `gui/prompt_list_presenter.py`
- Inspect: `gui/prompt_list_coordinator.py`
- Inspect: `gui/widgets/prompt_detail_widget.py`
- Inspect: `tests/test_prompt_list_model.py`
- Inspect: `tests/test_prompt_list_presenter.py`
- Inspect: `tests/test_prompt_detail_widget.py`
- Inspect: `tests/test_retrieval_cues_parity.py`
- Maintain: `docs/plans/2026-05-10-search-detail-reuse-confidence-roadmap.md`

**Steps:**
1. Inspect the active prompt-list and detail seams.
2. Confirm what cues are already shipped and should not be duplicated.
3. Identify one bounded hesitation seam where the result looks relevant but the next action is still less obvious than it should be.
4. Record the chosen seam under this task before implementation starts.

**Verification:**
- read the inspected files,
- search existing tests for the current cue wording,
- confirm the chosen seam is not already covered by shipped behavior/tests.

---

### Task 2: Add a failing focused test for one bounded confidence cue

**Status:** completed

**Objective:** Freeze one missing confidence/decision-support improvement before UI code changes.

**Selected v1 cue (frozen by RED test):**
- Surface: prompt-list / search-result seam only
- Trigger: title match path (`Matched in title`)
- Cue wording: `Likely reusable as-is`
- Reason for choosing this seam first: it is the smallest non-invasive handoff cue because title matches already have an explicit list-side reason cue, while the list still lacks any bounded preview of likely next-step confidence.
- Guardrail: this stays list-local and does not reuse full detail-view `Decision` or `Recommended next action` strings.

**RED evidence (confirmed):**
- Added focused test in `tests/test_prompt_list_model.py`:
  - `test_prompt_list_model_exposes_title_match_handoff_cue_for_immediate_reuse`
- Focused run fails as expected because `PromptListModel.HandoffCueRole` does not exist yet.
- Failure observed:
  - `AttributeError: type object 'PromptListModel' has no attribute 'HandoffCueRole'`

**Files:**
- Modify: one focused test file among:
  - `tests/test_prompt_list_model.py`
  - `tests/test_prompt_list_presenter.py`
  - `tests/test_prompt_detail_widget.py`
  - `tests/test_retrieval_cues_parity.py`
- Reference: the matching GUI seam chosen in Task 1

**Files:**
- Modify: one focused test file among:
  - `tests/test_prompt_list_model.py`
  - `tests/test_prompt_list_presenter.py`
  - `tests/test_prompt_detail_widget.py`
  - `tests/test_retrieval_cues_parity.py`
- Reference: the matching GUI seam chosen in Task 1

**Constraint:**
Pick only one cue family for v1.
Do not open several wording/UI ideas at once.

**Verification:**
Run only the focused test(s) for the chosen seam and confirm RED before implementation.

---

### Task 3: Implement the bounded confidence cue on the existing seam

**Status:** completed

**Objective:** Add the smallest operator-facing cue that reduces hesitation without expanding the retrieval model.

**Implemented v1 behavior:**
- Added a new list-local role in `gui/prompt_list_model.py`:
  - `HandoffCueRole`
- Current bounded rule:
  - when the active list-side reason is `Matched in title`,
  - expose `Likely reusable as-is`
- This is intentionally minimal:
  - no ranking change,
  - no persistence change,
  - no reuse of detail-view `Decision` / `Recommended next action` text,
  - no promotion into shared analytics.

**Verification (confirmed):**
- Focused GREEN:
  - `pytest tests/test_prompt_list_model.py::test_prompt_list_model_exposes_title_match_handoff_cue_for_immediate_reuse -v`
  - result: passed
- Nearby smoke pack:
  - `pytest tests/test_prompt_list_model.py tests/test_retrieval_cues_parity.py -q`
  - result: `20 passed`

**Files:**
- Modify: the single GUI seam chosen in Task 1
- Modify: the focused test file from Task 2
- Maybe modify: `tests/test_retrieval_cues_parity.py` if the cue must remain GUI-local

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

## Stage B — Detail-to-action confidence

### Task 4: Audit next-step wording in detail view against available actions

**Status:** completed

**Objective:** Verify whether detail-view guidance cleanly distinguishes reuse, refine, fork, and workspace-open actions.

**Chosen ambiguity seam (confirmed):**
- For prompts with `Decision: Reuse as-is` and no or thin run history, the detail surface currently recommends `Validate before reuse`.
- At the same time, the Quick Reuse area still exposes the concrete action buttons `Copy Prompt` and `Open in Workspace` as the immediate operator exits.
- This creates one bounded ambiguity seam: the next-action wording says what judgment is still needed, but does not tell the operator which available action best supports that validation-first move.
- In practice, the operator still has to infer whether `Validate before reuse` means copy the prompt directly, open it in Workspace first, or avoid both until more evidence exists.
- This is most visible on the `Reuse as-is` / limited-evidence path because the wording is calm but action selection remains slightly implicit.

**Audit notes (confirmed from code/tests):**
- `gui/workspace_history_controller.py` maps `Reuse as-is` to `Validate before reuse` when evidence is absent, stale, or limited.
- `gui/widgets/prompt_detail_widget.py` keeps `Copy Prompt` and `Open in Workspace` visible/enabled based on available text, but the bounded visible workspace handoff cue is template-specific only.
- `tests/test_workspace_history_controller.py` already locks multiple `Reuse as-is` -> `Validate before reuse` paths.
- `tests/test_prompt_detail_widget.py` locks button availability/tooltips, but does not yet lock a general non-template visible cue connecting validation-first wording to one preferred action surface.
- `tests/test_canonical_operator_path_parity.py` confirms the canonical path ends at `Copy Prompt` or `Open in Workspace`, reinforcing that this ambiguity is about action guidance rather than missing actions.

**Files:**
- Inspect: `gui/widgets/prompt_detail_widget.py`
- Inspect: nearby controller/dialog files that feed detail actions
- Inspect: `tests/test_prompt_detail_widget.py`
- Inspect: nearby tests covering workspace handoff or refinement cues

**Files:**
- Inspect: `gui/widgets/prompt_detail_widget.py`
- Inspect: nearby controller/dialog files that feed detail actions
- Inspect: `tests/test_prompt_detail_widget.py`
- Inspect: nearby tests covering workspace handoff or refinement cues

**Verification:**
- identify one ambiguity where the detail surface offers actions but the current wording/order leaves avoidable hesitation,
- confirm the ambiguity is not already intentionally covered by an existing cue.

---

### Task 5: Add failing focused test for detail-to-action confidence improvement

**Status:** completed

**Objective:** Lock one bounded detail-view improvement before changing widget/controller logic.

**Selected v1 improvement (frozen by RED test):**
- Surface: visible detail-side workspace handoff cue
- Trigger: plain prompt + `Decision: Reuse as-is` + `Recommended next action: Validate before reuse`
- Cue wording: `Open in Workspace before validating reuse.`
- Rationale: this is the smallest action-guidance improvement that resolves the ambiguity found in Task 4 without changing decision semantics, button labels, or adding a second workflow layer.

**RED evidence (confirmed):**
- Added focused test in `tests/test_prompt_detail_widget.py`:
  - `test_prompt_detail_widget_shows_workspace_validation_handoff_for_validate_before_reuse`
- Focused run fails because the current widget does not show any visible workspace handoff cue for this non-template validation-first path.
- Failure observed:
  - `assert _workspace_handoff_cue_label(widget).isVisible()` -> `False`

**Files:**
- Modify: `tests/test_prompt_detail_widget.py`
- Maybe modify: one nearby controller test if the cue comes from shared decision logic

**Files:**
- Modify: `tests/test_prompt_detail_widget.py`
- Maybe modify: one nearby controller test if the cue comes from shared decision logic

**Verification:**
Run the focused test and confirm RED.

---

### Task 6: Implement the detail-to-action confidence improvement

**Status:** completed

**Objective:** Make the next action easier to choose from the detail view without adding a new workflow layer.

**Implemented v1 behavior:**
- Extended the existing visible workspace handoff cue in `gui/widgets/prompt_detail_widget.py`.
- Current bounded rule:
  - if the prompt has reusable body text and the visible next-action summary is `Validate before reuse`,
  - show the reuse-area cue `Open in Workspace before validating reuse.`
- Existing template-specific cue remains intact:
  - `Open in Workspace to fill variables before reuse.`
- The widget now refreshes the visible workspace handoff cue when `update_next_action_summary(...)` changes the detail guidance, so the handoff remains aligned with the current action wording.

**Verification (confirmed):**
- Focused GREEN:
  - `pytest tests/test_prompt_detail_widget.py::test_prompt_detail_widget_shows_workspace_validation_handoff_for_validate_before_reuse -v`
  - result: passed
- Nearby smoke pack:
  - `pytest tests/test_prompt_detail_widget.py tests/test_workspace_history_controller.py -q`
  - result: `58 passed`

**Files:**
- Modify: `gui/widgets/prompt_detail_widget.py`
- Maybe modify: one nearby controller/helper file that already computes decision-support text
- Modify: focused tests from Task 5

**Files:**
- Modify: `gui/widgets/prompt_detail_widget.py`
- Maybe modify: one nearby controller/helper file that already computes decision-support text
- Modify: focused tests from Task 5

**Implementation targets:**
- keep the action guidance compact,
- avoid repetitive recommendation text,
- prefer explicit reuse vs refine clarity over generic encouragement,
- do not add new persistence or broad controller churn.

**Verification:**
Run focused detail-widget tests and one nearby smoke pack.

---

## Stage C — Ledger closure and docs sync

### Task 7: Sync the execution ledger after the bounded slice lands

**Status:** completed

**Objective:** Keep one unambiguous pointer chain after implementation.

**Implemented:**
- Stage A delivered one bounded search/result cue on the existing prompt-list seam:
  - `Matched in title` -> `Likely reusable as-is`
- Stage B delivered one bounded detail-to-action handoff cue on the existing detail seam:
  - `Validate before reuse` -> `Open in Workspace before validating reuse.`
- The execution ledger now records Task 1-6 as completed for this slice.

**Verified:**
- Focused and nearby tests for both seams pass.
- Ruff passes on touched implementation/test files.
- Direction pointer was later revised:
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - now points to `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md` as the active next-cycle execution ledger.
- No product-truth rewrite was needed in `docs/product-ssot.md`.
- No README change was required because the user-visible product posture did not change.

**Pointer-chain outcome:**
- `docs/plans/2026-04-25-roadmap-implementation-plan.md`
  -> points to
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  -> points to
- `docs/plans/2026-05-10-search-detail-reuse-confidence-roadmap.md`
- This file remains the active execution ledger for the completed bounded slice and the authoritative closure record for this cycle chunk.

**Next-cycle follow-up rule:**
- Do not leave `search/detail/reuse confidence cue v1` advertised as an open next slice anymore.
- Treat this slice as landed.
- The next bounded slice should be chosen explicitly in a later revision rather than implied by stale copy in this ledger.

**Files:**
- Modify: `docs/plans/2026-05-10-search-detail-reuse-confidence-roadmap.md`
- Maybe modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Maybe modify: `README.md` only if user-visible positioning changed
- Maybe modify: `docs/product-ssot.md` only if product truth changed
- Maybe modify: `docs/CHANGELOG.md` only if user-visible behavior changed or the ledger needs a short trace

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

## Current recommended next slice

**Status:** closed for this ledger

The bounded slice tracked here has landed:
- search/result confidence cue v1
- detail-to-action workspace handoff cue v1

Do not treat this section as an open implementation pointer anymore.
The next bounded slice must be selected in a later revision of the direction note and/or a new execution ledger.

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
- one bounded confidence improvement has shipped on an existing retrieval/detail seam,
- focused tests pass,
- nearby smoke tests pass,
- Ruff passes on touched files,
- the docs pointer chain remains unambiguous,
- no broader retrieval/workflow/product drift was introduced.
