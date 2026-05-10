# Detail Decision-Package Consistency Roadmap

- **Date:** 2026-05-10
- **Owner:** Hermes
- **Status:** proposed active bounded slice
- **Depends on:**
  - `docs/product-ssot.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`
  - `docs/plans/2026-05-10-detail-comparable-evidence-provenance-clarity-roadmap.md`

## Why this slice

The canonical SSOT now prioritizes:
- retrieval-to-action confidence,
- inspect/detail as a clear decision surface,
- reuse / refine / fork operating path.

Recent bounded detail slices already improved:
- next-step closure for validate / baseline / edit / inspect / fork,
- thin-evidence provenance readability,
- comparable-evidence provenance readability.

The next bounded gap is no longer a single wording cue.
It is package consistency:
- the shared detail seam now has separate contracts for `Decision`, `Decision provenance`, `Recommended next action`, and `Next step`,
- but there is not yet one focused contract proving that these surfaces form a coherent operator-facing package when shown together,
- especially on the validate path, which is a strong representative case for inspect/detail decision support.

This makes the next best bounded product question:

**When the detail surface shows decision, provenance, next action, and next step together, does it present one coherent package instead of four isolated cues?**

## Boundaries

This slice must stay bounded to one existing seam:
- `gui/widgets/prompt_detail_widget.py`
- `tests/test_prompt_detail_widget.py`

Do not:
- redesign retrieval ranking,
- widen prompt-list logic,
- add new persistence,
- rewrite controllers,
- add new panels,
- reopen already shipped single-cue wording slices unless required by the new package-level contract,
- broaden workbench or prompt-chain workflows.

## Proposed task bundle

### Task 1 — Verify the current package-level gap
**Status:** completed

**Objective:** Prove with focused evidence what the shared detail widget currently guarantees when decision, provenance, next action, and next step are all present together, and what it does not yet guarantee.

**Implemented audit:**
- re-read `gui/widgets/prompt_detail_widget.py`,
- re-read `tests/test_prompt_detail_widget.py`,
- confirmed there is no focused package-level contract yet for a representative validate-path state,
- confirmed existing tests remain mostly per-cue rather than package-level.

**Verified result:**
- the shared detail seam already exposes all four package components,
- the validate path is a good representative seam because it already has decision, next action, and next step coverage,
- but there is still no single bounded contract proving that decision, provenance, next action, and next step form one coherent operator-facing package when rendered together,
- no code changes were made in this audit step.

### Task 2 — Add one RED test for decision-package consistency
**Status:** completed

**Objective:** Capture one stronger bounded contract for a coherent detail-side decision package.

**Implemented:**
- Added one focused validate-path package test in `tests/test_prompt_detail_widget.py`.
- The new test keeps scope local to the shared detail widget and exercises one representative state where all four package elements are present together.
- The bounded contract now verifies that the validate path exposes:
  - `Decision`
  - `Decision provenance`
  - `Recommended next action`
  - `Next step`
  as one coherent visible package.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> RED before test adjustment because the first draft assumed plain-text label rendering rather than the widget's existing HTML label formatting.
- The failure stayed local to the shared detail seam and exposed the actual rendering contract that the package test should follow.

### Task 3 — Implement the smallest GREEN fix in the detail widget
**Status:** completed

**Objective:** Add the smallest bounded fix that satisfies the new package-level contract.

**Implemented:**
- No widget code change was required.
- The RED step showed that the real gap was missing package-level coverage, not missing widget behavior.
- The minimal GREEN outcome was therefore to align the new test with the widget's existing HTML label formatting contract while keeping the package-level assertions intact.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> GREEN (`41 passed`)
- the validate path now has one explicit package-level coverage contract on the shared detail seam.

### Task 4 — Verify and close docs pointers
**Status:** completed

**Objective:** Close the slice with focused verification and doc notes.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> `41 passed`
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`

**Docs closure:**
- this ledger now records Task 1–4 through implementation and verification,
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` already points to this file as the active bounded execution ledger,
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` already points to this file in the next-cycle pointer chain,
- `docs/product-ssot.md` stays unchanged because product truth did not change.

## Current recommended next slice

**Status:** delivered for the current v1 seam

The bounded `detail decision-package consistency` slice is now landed on the shared detail seam:
- `tests/test_prompt_detail_widget.py` now has one explicit validate-path package contract covering decision, provenance, next action, and next step together,
- focused widget verification and Ruff checks pass,
- the seam now has package-level coverage without broadening the product surface.

**Next successor to choose explicitly:**
- either one more bounded inspect/detail consistency pass if a real hesitation seam remains,
- or move to the next reuse/refine/fork continuity seam if package-level detail confidence is now good enough.

Selection rule for the next successor:
- do not reopen the shipped validate package contract unless a real behavior gap appears,
- prefer the next smallest operator-facing seam over another wording-only micro-slice,
- avoid any slice that would require controller rewrites, persistence changes, or broader workflow expansion.

## Definition of done

This slice is done only when:
- one focused RED->GREEN test proves the detail decision-package consistency contract,
- the shared detail widget improves package-level clarity in one bounded way,
- shipped validate/baseline/edit/inspect/fork handoff behavior remains intact,
- shipped limited/comparable provenance wording remains intact,
- `pytest tests/test_prompt_detail_widget.py -q` passes,
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` passes,
- docs pointer chain names this file as the active bounded execution ledger,
- no broader retrieval/detail/workflow drift is introduced.
