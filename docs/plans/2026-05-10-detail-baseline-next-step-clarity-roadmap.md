# Detail Baseline Next-Step Clarity Roadmap

- **Date:** 2026-05-10
- **Owner:** Hermes
- **Status:** proposed active bounded slice
- **Depends on:**
  - `docs/product-ssot.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`
  - `docs/plans/2026-05-10-detail-inspect-next-step-clarity-roadmap.md`

## Why this slice

The canonical SSOT now prioritizes:
- retrieval-to-action confidence,
- inspect/detail as a clear decision surface,
- reuse / refine / fork operating path.

Recent bounded detail slices already improved:
- `Validate before reuse` -> `Next step: Open in Workspace before validating reuse.`,
- `Edit Prompt before reuse` -> `Next step: Edit Prompt before reuse.`,
- `Inspect before reuse` -> `Next step: Review prompt details before reusing.`,
- `Fork before editing` -> `Next step: Fork Prompt before editing.`

The remaining bounded gap is now narrower:
- the shared detail widget already renders baseline-path wording,
- `Decision: Keep baseline` and `Recommended next action: Prefer baseline before reuse` are covered in tests,
- but there is no equivalent bounded `Next step` handoff for the baseline path on the shared detail surface.

This makes the next best bounded product question:

**When detail guidance says `Keep baseline`, should the widget expose one compact `Next step` cue that closes the baseline path as clearly as validate/edit/inspect/fork?**

## Boundaries

This slice must stay bounded to one existing seam:
- `gui/widgets/prompt_detail_widget.py`
- `tests/test_prompt_detail_widget.py`

Do not:
- redesign retrieval ranking,
- widen prompt-list logic,
- add new persistence,
- rewrite controllers,
- broaden workbench or prompt-chain workflows,
- reopen already shipped validate/edit/inspect/fork handoff cues.

## Proposed task bundle

### Task 1 — Verify the missing baseline-next-step contract
**Status:** completed

**Objective:** Prove with focused evidence that the shared detail widget currently lacks a dedicated bounded `Next step` handoff for the baseline path.

**Implemented audit:**
- re-read `gui/widgets/prompt_detail_widget.py`,
- re-read `tests/test_prompt_detail_widget.py`,
- confirmed the baseline path currently stops at:
  - `Decision: Keep baseline`
  - `Recommended next action: Prefer baseline before reuse`,
- confirmed `_resolve_workspace_handoff_cue(...)` has no equivalent bounded branch for that baseline path.

**Verified result:**
- the baseline-gap is real on the shared detail seam,
- the baseline decision path is weaker than the already shipped validate/edit/inspect/fork handoff paths,
- no code changes were made in this audit step.

### Task 2 — Add one RED test for baseline detail handoff
**Status:** completed

**Objective:** Capture the desired contract in one focused widget test before implementation.

**Implemented:**
- Added focused coverage in `tests/test_prompt_detail_widget.py` for the baseline detail path.
- The test displays a prompt with real context, sets `Decision: Keep baseline` and `Recommended next action: Prefer baseline before reuse`, and requires one visible `Next step:` handoff cue.
- The bounded wording contract now expects: `Reuse the baseline prompt.`

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> RED before implementation (`1 failed`) because the baseline path did not expose a visible detail-side handoff cue.
- The RED failure stayed local to the shared detail seam.

### Task 3 — Implement the smallest GREEN fix in the detail widget
**Status:** completed

**Objective:** Add the smallest bounded detail-side handoff branch that satisfies the RED test.

**Implemented:**
- Extended `_resolve_workspace_handoff_cue(...)` in `gui/widgets/prompt_detail_widget.py` with one baseline branch:
  - `Prefer baseline before reuse` -> `Next step: Reuse the baseline prompt.`
- Kept shipped behavior for validate/edit/inspect/fork/template-variable paths intact.
- Refactored the local context-presence guard into one bounded `has_context` helper variable to keep the resolver readable while preserving behavior.
- Kept the fix bounded to the shared detail widget without widening controller or workflow scope.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> GREEN (`38 passed`)
- the baseline path now closes on the shared detail seam with one compact `Next step` cue.

### Task 4 — Verify and close docs pointers
**Status:** completed

**Objective:** Close the slice with focused verification and doc pointer updates.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> `38 passed`
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`

**Docs closure:**
- this ledger now records Task 1–4 through implementation and verification,
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` already points to this file as the active bounded execution ledger,
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` already points to this file in the next-cycle pointer chain,
- `docs/product-ssot.md` stays unchanged because product truth did not change.

## Current recommended next slice

**Status:** delivered for the current v1 seam

The bounded `detail baseline next-step clarity` slice is now landed on the shared detail seam:
- `tests/test_prompt_detail_widget.py` now covers the baseline detail handoff path,
- `gui/widgets/prompt_detail_widget.py` now shows one compact `Next step` cue for `Prefer baseline before reuse`,
- focused widget verification and Ruff checks pass.

**Next successor to choose explicitly:**
- either one small follow-up on detail-surface consistency,
- or move to the next bounded reuse/refine/fork continuity gap if that is now the better product question.

Selection rule for the next successor:
- do not reopen the shipped baseline handoff wording seam itself,
- prefer the smallest remaining hesitation seam on inspect/detail or reuse/refine/fork continuity,
- avoid any slice that would require controller rewrites, persistence changes, or broader workflow expansion.

## Definition of done

This slice is done only when:
- one focused RED->GREEN test proves the baseline handoff contract,
- the shared detail widget exposes one bounded baseline-side `Next step` cue,
- existing validate/edit/inspect/fork/template-variable handoff behavior remains intact,
- `pytest tests/test_prompt_detail_widget.py -q` passes,
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` passes,
- docs pointer chain names this file as the active bounded execution ledger,
- no broader retrieval/detail/workflow drift is introduced.
