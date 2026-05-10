# Detail Inspect Next-Step Clarity Roadmap

- **Date:** 2026-05-10
- **Owner:** Hermes
- **Status:** proposed active bounded slice
- **Depends on:**
  - `docs/product-ssot.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`
  - `docs/plans/2026-05-10-detail-refine-fork-action-clarity-roadmap.md`
  - `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md`

## Why this slice

The canonical SSOT now prioritizes:
- retrieval-to-action confidence,
- inspect/detail as a clear decision surface,
- reuse / refine / fork operating path.

Recent bounded slices already improved:
- prompt-list handoff cues such as `Likely reusable as-is` and `Inspect before reuse`,
- detail-side `Next step` handoff cues for:
  - `Validate before reuse`,
  - `Edit Prompt before reuse`,
  - `Fork before editing`.

The remaining bounded gap is now narrower:
- `Inspect before reuse` exists as a prompt-list/operator cue,
- but the shared detail widget does not yet expose a corresponding bounded `Next step` handoff on the inspect surface,
- so the inspect recommendation is actionable at list level but not fully closed on the detail surface.

This makes the next best bounded product question:

**When detail-side guidance effectively says inspect before reuse, should the widget expose one compact `Next step` cue that closes the inspect path without widening the workflow?**

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
- reopen already shipped validate/edit/fork handoff cues.

## Proposed task bundle

### Task 1 — Verify the missing inspect-next-step contract
**Status:** completed

**Objective:** Prove with focused evidence that the shared detail widget currently lacks a dedicated bounded `Next step` handoff for the inspect-first path.

**Implemented audit:**
- re-read `gui/widgets/prompt_detail_widget.py`,
- re-read `tests/test_prompt_detail_widget.py`,
- confirmed `_resolve_workspace_handoff_cue(...)` currently handles only:
  - `Validate before reuse`,
  - `Edit Prompt before reuse`,
  - `Fork before editing`,
  - template-variable fallback,
- confirmed there is no equivalent bounded branch for `Inspect before reuse`.

**Verified result:**
- the inspect-first gap is real on the shared detail seam,
- the list-side/operator recommendation is not yet closed by a detail-side `Next step` cue,
- no code changes were made in this audit step.

### Task 2 — Add one RED test for inspect-first detail handoff
**Status:** completed

**Objective:** Capture the desired contract in one focused widget test before implementation.

**Implemented:**
- Added focused coverage in `tests/test_prompt_detail_widget.py` for the inspect-first detail path.
- The test displays a prompt with real context, sets both `Decision` and `Recommended next action` to `Inspect before reuse`, and requires one visible `Next step:` handoff cue.
- The bounded wording contract now expects: `Review prompt details before reusing.`

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> RED before implementation (`1 failed`) because the inspect-first path did not expose a visible workspace/detail handoff cue.
- The RED failure stayed local to the shared detail seam.

### Task 3 — Implement the smallest GREEN fix in the detail widget
**Status:** completed

**Objective:** Add the smallest bounded detail-side handoff branch that satisfies the RED test.

**Implemented:**
- Extended `_resolve_workspace_handoff_cue(...)` in `gui/widgets/prompt_detail_widget.py` with one inspect-first branch:
  - `Inspect before reuse` -> `Next step: Review prompt details before reusing.`
- Kept shipped behavior for validate/edit/fork/template-variable paths unchanged.
- Kept the fix bounded to the shared detail widget without widening controller or workflow scope.
- Adjusted the duplicate-next-action suppression rule so `Inspect before reuse`, like the already shipped fork path, can remain visible on the detail surface when that explicit duplication is needed to support the bounded handoff cue.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> GREEN (`37 passed`)
- inspect-first handoff now closes on the shared detail seam with one compact `Next step` cue.

### Task 4 — Verify and close docs pointers
**Status:** completed

**Objective:** Close the slice with focused verification and doc pointer updates.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> `37 passed`
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`

**Docs closure:**
- this ledger now records Task 1–4 through implementation and verification,
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` already points to this file as the active bounded execution ledger,
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` already points to this file in the next-cycle pointer chain,
- `docs/product-ssot.md` stays unchanged because product truth did not change.

## Current recommended next slice

**Status:** delivered for the current v1 seam

The bounded `detail inspect next-step clarity` slice is now landed on the shared detail seam:
- `tests/test_prompt_detail_widget.py` now covers the inspect-first detail handoff path,
- `gui/widgets/prompt_detail_widget.py` now shows one compact `Next step` cue for `Inspect before reuse`,
- focused widget verification and Ruff checks pass.

**Next successor to choose explicitly:**
- either one small follow-up on detail-surface consistency,
- or move to the next bounded reuse/refine/fork continuity gap if that is now the better product question.

Selection rule for the next successor:
- do not reopen the shipped inspect-first handoff wording seam itself,
- prefer the smallest remaining hesitation seam on inspect/detail or reuse/refine/fork continuity,
- avoid any slice that would require controller rewrites, persistence changes, or broader workflow expansion.

## Definition of done

This slice is done only when:
- one focused RED->GREEN test proves the inspect-first handoff contract,
- the shared detail widget exposes one bounded inspect-side `Next step` cue,
- existing validate/edit/fork/template-variable handoff behavior remains intact,
- `pytest tests/test_prompt_detail_widget.py -q` passes,
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` passes,
- docs pointer chain names this file as the active bounded execution ledger,
- no broader retrieval/detail/workflow drift is introduced.
