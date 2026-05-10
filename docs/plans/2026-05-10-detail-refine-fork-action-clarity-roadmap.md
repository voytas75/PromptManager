# PromptManager Detail Refine/Fork Action Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce operator hesitation on refine-vs-fork decisions directly on the existing detail/action seam, so the user can tell the next concrete move more confidently without introducing a new workflow layer, controller rewrite, or persistence model.

**Architecture:** This roadmap stays inside the current prompt detail surface and its immediate action affordances. It should improve action clarity by tightening the relationship between existing decision-support cues (`Decision`, provenance, `Recommended next action`) and the already-shipped actions (`Edit Prompt`, `Open in Workspace`, `Copy Prompt`, `Promote Draft`) rather than adding a broader refinement workflow. Each slice should remain bounded, testable, and local to the operator flow: inspect -> judge next move -> refine or fork safely.

**Tech Stack:** Python 3.13, PySide6 GUI seams, PromptManager detail/presenter/controller surfaces, pytest, Ruff, existing product SSOT docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Previous delivered bounded ledger: `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/product-ssot.md` keeps inspection clarity, reuse/refine, and version/fork clarity inside the core product loop.
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` now records this detail refine/fork slice as a delivered previous slice; the next bounded successor must be chosen explicitly.
- `gui/widgets/prompt_detail_widget.py` already exposes bounded decision-support surfaces:
  - `Decision`
  - decision provenance
  - `Recommended next action`
- the detail widget already exposes immediate action surfaces:
  - `Edit Prompt`
  - `Copy Prompt`
  - `Open in Workspace`
  - `Promote Draft`
- focused detail-widget tests already cover wording and visibility for current decision and next-action cues.

So the next useful move is:

**asset-first core -> detail-side refine/fork action clarity on the existing inspection/action seam**

---

## Scope guardrails

This roadmap may improve only:
- one bounded hesitation seam on the existing detail/action surface,
- compact action guidance that clarifies refine vs fork intent,
- local wording/rendering/handoff improvements that stay on current detail seams,
- focused tests proving the cue/action contract.

This roadmap must not introduce:
- a new workspace workflow layer,
- a controller or presenter rewrite,
- new persistence or lineage schema,
- a broader refinement engine,
- dashboard/analytics detours,
- duplicated product semantics across multiple new surfaces.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- detail view already exposes compact `Decision`, provenance, and `Recommended next action` cues,
- quick-reuse actions already exist on the detail surface,
- shared detail action row already exists for edit/favorite/promote behavior,
- prompt-list search-result action clarity is already delivered as the previous bounded slice,
- the current product direction note now frames this seam as the active bounded slice.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> when the detail view suggests refinement-oriented handling, how do we reduce hesitation between “edit this prompt”, “fork before editing”, and “open it in workspace first” without introducing a larger refinement workflow?

Preferred answer shape:
- one bounded refine/fork hesitation seam,
- one clearer detail-side next-move expectation,
- no workflow expansion,
- all on the current detail/action surface.

---

## Stage A — Detail refine/fork action clarity

### Task 1: Audit the current refine/fork hesitation seam

**Status:** completed

**Objective:** Confirm the smallest still-ambiguous operator decision on the existing detail/action surface before implementation starts.

**Chosen bounded hesitation seam (confirmed):**
- The detail surface already exposes `Decision` and `Recommended next action` cues, and the widget already renders both `Edit Prompt` and `Fork Prompt` as separate actions.
- Existing focused tests cover decision/next-action wording such as:
  - `Refine before reuse`
  - `Reuse as-is`
  - `Validate before reuse`
- Existing action/tooling coverage outside the widget already distinguishes fork lineage semantics:
  - `Fork Prompt` is documented/tested as `Create a fork linked to this prompt and open it for editing.`
- The remaining bounded ambiguity is therefore not missing fork capability; it is missing **detail-local handoff clarity** when the decision becomes refinement-oriented.
- On the current detail widget seam, a prompt can show `Decision: Refine before reuse` while the visible action row still leaves the operator to infer whether the concrete next move should be `Edit Prompt` or `Fork Prompt`.
- No existing detail-widget cue currently bridges that gap the way validation-oriented flows already do with the visible workspace handoff cue:
  - `Next step: Open in Workspace before validating reuse.`
- The selected v1 direction is therefore **one detail-local next-step handoff cue for the refine/fork seam**, without adding a new workflow layer and without rewriting controller/persistence behavior.

**Audit notes (confirmed from code/tests):**
- `gui/widgets/prompt_detail_widget.py` already declares and renders separate actions/signals for:
  - `edit_requested`
  - `fork_requested`
  - `open_in_workspace_requested`
- `gui/widgets/prompt_detail_widget.py` already has one precedent for bounded handoff guidance on this same surface:
  - `_resolve_workspace_handoff_cue()` maps `Validate before reuse` to a visible `Next step` cue.
- `tests/test_prompt_detail_widget.py` already locks current decision/next-action rendering and duplicate-hiding behavior, but does not yet assert any visible handoff cue for `Refine before reuse` or any fork-oriented next step.
- `tests/test_prompt_actions_controller.py` already locks the meaning of `Fork Prompt` as lineage-preserving editing, so the missing seam is not terminology definition; it is surfacing the right next move at detail time.
- Because the fork semantics already exist elsewhere, the smallest bounded seam is to reuse the existing detail-side handoff pattern rather than inventing a new action family.

**Files:**
- Inspect: `gui/widgets/prompt_detail_widget.py`
- Inspect: `tests/test_prompt_detail_widget.py`
- Inspect: `tests/test_prompt_actions_controller.py`
- Maybe inspect: nearby presenter/controller files that feed detail cues
- Maintain: `docs/plans/2026-05-10-detail-refine-fork-action-clarity-roadmap.md`

**Implemented:**
- Audited the detail widget, focused widget tests, and nearby action-semantics tests.
- Confirmed the smallest uncovered hesitation seam is a missing visible handoff from `Refine before reuse` to one concrete action expectation on the existing detail surface.
- Chose the v1 seam as a detail-local `Next step` cue rather than broader workflow or lineage changes.

**Verified:**
- Read:
  - `gui/widgets/prompt_detail_widget.py`
  - `tests/test_prompt_detail_widget.py`
- Searched nearby fork/edit semantics in:
  - `tests/test_prompt_actions_controller.py`
- Confirmed no existing detail-widget test or shipped widget cue already covers refine/fork handoff visibility.

---

### Task 2: Add a failing focused test for the chosen detail-side cue gap

**Status:** completed

**Objective:** Freeze one missing detail/action clarity improvement before UI code changes.

**Constraint:**
Pick only one cue family or handoff ambiguity for v1.
Do not open several competing detail ideas at once.

**Implemented:**
- Added one focused RED test in `tests/test_prompt_detail_widget.py` for the chosen detail-local handoff seam.
- The new test freezes the expected v1 behavior:
  - when the detail surface shows `Decision: Refine before reuse`
  - and a real prompt is displayed,
  - the visible `Next step` cue should surface:
    - `Edit Prompt before reuse.`
- Kept the test on the existing widget-local handoff surface (`promptWorkspaceHandoffCue`) instead of opening a new UI surface.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py::test_prompt_detail_widget_shows_refine_handoff_cue_for_edit_path -q` -> RED
- Failure confirms the current widget does **not** yet show a visible refine-path handoff cue (`promptWorkspaceHandoffCue` stayed hidden).

**Files:**
- Modify: `tests/test_prompt_detail_widget.py`
- Reference: `gui/widgets/prompt_detail_widget.py`

**Verification:**
Run only the focused test(s) for the chosen seam and confirm RED before implementation.

---

### Task 3: Implement the bounded detail-side refine/fork clarity cue

**Status:** completed

**Objective:** Add the smallest operator-facing cue or handoff improvement that reduces refine/fork hesitation without expanding the product model.

**Implemented:**
- Extended the existing detail-local handoff seam in `gui/widgets/prompt_detail_widget.py` instead of adding a new surface.
- Kept the already-shipped validation path unchanged:
  - `Recommended next action: Validate before reuse` -> `Next step: Open in Workspace before validating reuse.`
- Added one bounded refine-path handoff:
  - `Recommended next action: Edit Prompt before reuse` -> `Next step: Edit Prompt before reuse.`
- Reused the existing visible `promptWorkspaceHandoffCue` surface for this cue so the change stays widget-local and compact.
- Ensured the handoff cue is recomputed when decision state changes, not only on initial `display_prompt()`.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py::test_prompt_detail_widget_shows_refine_handoff_cue_for_edit_path -q` -> `1 passed`
- `pytest tests/test_prompt_detail_widget.py tests/test_prompt_actions_controller.py -q` -> `47 passed`
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`

**Files:**
- Modify: `gui/widgets/prompt_detail_widget.py`
- Modify: `tests/test_prompt_detail_widget.py`
- Reference: nearby action semantics remain covered in `tests/test_prompt_actions_controller.py`

**Implementation targets:**
- keep the cue compact,
- prefer existing terminology over new product language,
- avoid creating a new workflow surface,
- preserve current persistence and lineage behavior.

**Verification:**
Run the focused tests for the changed seam, plus one nearby smoke pack.

---

### Task 4: Sync the execution ledger after the bounded slice lands

**Status:** completed

**Objective:** Keep one unambiguous pointer chain after the new slice becomes active or lands.

**Implemented:**
- Re-read the active detail refine/fork ledger, the direction note, and the umbrella implementation plan after the Task 3 code/test closure.
- Confirmed the active pointer chain remains consistent:
  - strategic direction: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - active bounded execution ledger: `docs/plans/2026-05-10-detail-refine-fork-action-clarity-roadmap.md`
  - umbrella pointer: `docs/plans/2026-04-25-roadmap-implementation-plan.md`
- Confirmed the direction note marked at closure time:
  - Candidate 1 as `delivered previous slice`
  - Candidate 2 as `active current slice` before the doc-closure pass moved it to delivered state.
- Confirmed no README or `docs/product-ssot.md` patch is needed for this slice because product truth did not change; only one bounded detail-local handoff cue shipped.
- Updated this ledger so the implementation is recorded through Task 4 with explicit closure notes.

**Verified:**
- re-read:
  - `docs/plans/2026-05-10-detail-refine-fork-action-clarity-roadmap.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`
- confirmed the direction note points to `docs/plans/2026-05-10-detail-refine-fork-action-clarity-roadmap.md` as the current chosen bounded slice.
- confirmed the umbrella plan points to the same file as the active bounded execution ledger.
- confirmed there is no competing `active current slice` wording left for the prior delivered prompt-list slice.

**Files:**
- Modify: `docs/plans/2026-05-10-detail-refine-fork-action-clarity-roadmap.md`
- Maybe modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Maybe modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`
- Maybe modify: `README.md` only if user-visible positioning changes

## Current recommended next slice

**Status:** delivered for the current v1 seam

The bounded `detail refine/fork action clarity v1` slice is now landed on the active detail/action seam:
- a focused RED->GREEN test now covers the refine-path handoff expectation in `tests/test_prompt_detail_widget.py`,
- `gui/widgets/prompt_detail_widget.py` now shows one compact visible `Next step` cue for the refine edit path,
- focused and nearby smoke tests pass.

**Next successor to choose explicitly:**
- either one small follow-up on the same detail/action confidence seam,
- or move to another pending candidate from `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`.

Selection rule for the next successor:
- do not reopen the shipped v1 refine handoff cue itself,
- prefer the smallest remaining hesitation seam,
- avoid any slice that would require a new workflow layer, controller rewrite, or persistence changes.

---

## Update workflow after each implemented slice
