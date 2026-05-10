# Detail Limited-Evidence Provenance Clarity Roadmap

- **Date:** 2026-05-10
- **Owner:** Hermes
- **Status:** proposed active bounded slice
- **Depends on:**
  - `docs/product-ssot.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`
  - `docs/plans/2026-05-10-detail-baseline-next-step-clarity-roadmap.md`

## Why this slice

The canonical SSOT now prioritizes:
- retrieval-to-action confidence,
- inspect/detail as a clear decision surface,
- reuse / refine / fork operating path.

Recent bounded detail slices already improved operator-facing next-step closure for:
- `Validate before reuse`,
- `Prefer baseline before reuse`,
- `Edit Prompt before reuse`,
- `Inspect before reuse`,
- `Fork before editing`.

The remaining bounded gap is now more about evidence clarity than missing actions:
- `tests/test_prompt_detail_widget.py` already proves that `Based on limited run evidence` can be rendered,
- `gui/widgets/prompt_detail_widget.py` currently treats provenance as a plain label with no extra bounded readability contract,
- there is not yet a focused test proving that thin-evidence provenance remains readable and visibly connected to the current decision package on the detail surface.

This makes the next best bounded product question:

**When a decision rests on limited run evidence, does the shared detail surface communicate that uncertainty clearly enough to support the operator decision without widening the workflow?**

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
- reopen already shipped next-step cue slices,
- broaden workbench or prompt-chain workflows.

## Proposed task bundle

### Task 1 — Verify the limited-evidence provenance gap
**Status:** completed

**Objective:** Prove with focused evidence what the shared detail widget currently guarantees for limited-evidence provenance, and what it does not yet guarantee.

**Implemented audit:**
- re-read `gui/widgets/prompt_detail_widget.py`,
- re-read `tests/test_prompt_detail_widget.py`,
- confirmed current coverage only proves that `Based on limited run evidence` renders as visible text,
- confirmed there is no stronger bounded contract yet for how limited-evidence provenance should read alongside the decision package.

**Verified result:**
- the thin-evidence gap is real on the shared detail seam,
- provenance is currently treated as a plain visible label,
- there is still no focused contract that makes limited-evidence wording more operator-readable,
- no code changes were made in this audit step.

### Task 2 — Add one RED test for limited-evidence provenance clarity
**Status:** completed

**Objective:** Capture one stronger bounded readability contract before implementation.

**Implemented:**
- Added focused coverage in `tests/test_prompt_detail_widget.py` for the thin-evidence provenance path.
- The new test keeps scope local to the shared detail widget and requires limited-evidence provenance to read as an explicit cautionary note.
- The bounded wording contract now expects the provenance text to start with `Note:` and include `Based on limited run evidence.`

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> RED before implementation (`1 failed`) because the existing provenance label rendered only the raw baseline wording.
- The RED failure stayed local to the shared detail seam.

### Task 3 — Implement the smallest GREEN fix in the detail widget
**Status:** completed

**Objective:** Add the smallest bounded detail-side readability fix that satisfies the RED test.

**Implemented:**
- Extended `update_decision_provenance_summary(...)` in `gui/widgets/prompt_detail_widget.py` with one bounded wording rule:
  - `Based on limited run evidence` -> `Note: Based on limited run evidence.`
- Kept the change local to the shared detail widget.
- Preserved shipped action handoff behavior and did not introduce a second provenance system.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> GREEN (`39 passed`)
- limited-evidence provenance now reads as an explicit cautionary note on the detail surface.

### Task 4 — Verify and close docs pointers
**Status:** completed

**Objective:** Close the slice with focused verification and doc notes.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> `39 passed`
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`

**Docs closure:**
- this ledger now records Task 1–4 through implementation and verification,
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` already points to this file as the active bounded execution ledger,
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` already points to this file in the next-cycle pointer chain,
- `docs/product-ssot.md` stays unchanged because product truth did not change.

## Current recommended next slice

**Status:** delivered for the current v1 seam

The bounded `detail limited-evidence provenance clarity` slice is now landed on the shared detail seam:
- `tests/test_prompt_detail_widget.py` now covers the limited-evidence provenance cautionary wording contract,
- `gui/widgets/prompt_detail_widget.py` now formats thin-evidence provenance as `Note: Based on limited run evidence.`,
- focused widget verification and Ruff checks pass.

**Next successor to choose explicitly:**
- either one more small detail-surface consistency pass,
- or move to the next bounded reuse/refine/fork continuity gap if that is now the stronger product question.

Selection rule for the next successor:
- do not reopen the shipped limited-evidence wording seam itself,
- prefer the smallest remaining hesitation seam on inspect/detail or reuse/refine/fork continuity,
- avoid any slice that would require controller rewrites, persistence changes, or broader workflow expansion.

## Definition of done

This slice is done only when:
- one focused RED->GREEN test proves the limited-evidence provenance clarity contract,
- the shared detail widget improves thin-evidence readability in one bounded way,
- shipped validate/baseline/edit/inspect/fork handoff behavior remains intact,
- `pytest tests/test_prompt_detail_widget.py -q` passes,
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` passes,
- docs pointer chain names this file as the active bounded execution ledger,
- no broader retrieval/detail/workflow drift is introduced.
