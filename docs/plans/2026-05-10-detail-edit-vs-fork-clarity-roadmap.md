# PromptManager Detail Edit-vs-Fork Clarity Roadmap

- **Date:** 2026-05-10
- **Owner:** Hermes
- **Status:** proposed active bounded slice
- **Depends on:**
  - `docs/product-ssot.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-05-10-detail-copy-vs-workspace-action-clarity-roadmap.md`
- **Target seam:**
  - `gui/widgets/prompt_detail_widget.py`
  - `tests/test_prompt_detail_widget.py`
- **Out of scope:** controller rewrites, new lineage model, workspace workflow expansion, prompt actions controller changes, broader refinement engine behavior.

---

## Why this slice exists

The current detail seam already ships:
- `Edit Prompt before reuse` -> `Next step: Edit Prompt before reuse.`
- `Fork before editing` -> `Next step: Fork Prompt before editing.`
- direct reuse -> `Next step: Copy Prompt for direct reuse.`

So the remaining bounded gap is no longer a missing cue on either path by itself.
The remaining hesitation seam is whether the detail surface helps the operator tell **why** the fork path is different from the edit path when both actions are visible.

This slice stays bounded by clarifying only one edit-vs-fork decision boundary on the existing detail seam.

---

## Product question

> When both `Edit Prompt` and `Fork Prompt` are available, can the detail surface make the fork-preserving path more explicit without adding a new workflow model?

Preferred answer shape:
- keep both existing actions,
- keep current action labels and controller semantics,
- improve only one detail-local decision boundary,
- avoid duplicating controller/tooltips into a new explanation layer.

---

## Task 1 — Verify the current edit-vs-fork gap
**Status:** completed

**Objective:** Confirm the smallest real remaining ambiguity between edit and fork paths on the current detail seam.

**Confirmed gap:**
- the detail widget already exposes separate next-step cues for edit and fork paths, but the fork path still does not explicitly tell the operator why that path is safer when the goal is to preserve the current prompt while making changes.

**Audit notes:**
- `gui/widgets/prompt_detail_widget.py` already renders:
  - `Edit Prompt before reuse.` for the refine path,
  - `Fork Prompt before editing.` for the fork path.
- `tests/test_prompt_detail_widget.py` already proves both cues independently.
- `Fork Prompt` semantics are already established elsewhere as lineage-preserving, but that distinction is not yet surfaced in the detail-local handoff cue itself.
- The smallest remaining bounded seam is therefore not another new path, but a slightly stronger fork-path expectation that explains preservation intent on the existing cue surface.

---

## Task 2 — Add one RED test for the chosen edit-vs-fork contract
**Status:** completed

**Objective:** Freeze one missing operator-facing boundary before implementation.

**Implemented:**
- tightened the existing fork-path widget test in `tests/test_prompt_detail_widget.py` to require a stronger preservation-aware cue:
  - `Next step: Fork Prompt to preserve the current version before editing.`
- RED confirmed the old wording only said `Fork Prompt before editing.` and did not surface the preservation boundary explicitly.

---

## Task 3 — Implement the smallest GREEN fix on the detail seam
**Status:** completed

**Objective:** Land one compact edit-vs-fork boundary cue without expanding the workflow model.

**Implemented:**
- kept both actions and all controller semantics unchanged,
- strengthened only the fork-path handoff text in `gui/widgets/prompt_detail_widget.py`:
  - from `Fork Prompt before editing.`
  - to `Fork Prompt to preserve the current version before editing.`
- left the edit path unchanged:
  - `Edit Prompt before reuse.`

**Why this is minimal:**
- no new UI surface,
- no controller changes,
- no lineage model changes,
- only one stronger fork/edit boundary on the existing handoff cue.

---

## Task 4 — Verify and assess stage closure
**Status:** completed

**Objective:** Verify the bounded slice and record whether the current reuse/refine/fork stage looks closure-ready after this delivery.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> `42 passed`
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`

**Stage-closure assessment:**
- copy-vs-workspace clarity: delivered
- edit-path refine cue: delivered
- fork-path preservation-aware cue: delivered
- remaining stage-wide gaps for `reuse / refine / fork operating path` are now **do weryfikacji**, but there is no longer an obvious detail-local hesitation seam of the same priority on the current widget surface.
- practical read: the current stage looks closure-ready for commit unless a new concrete continuity gap is identified outside this seam.

---

## Definition of done

This slice is done when:
- one real edit-vs-fork ambiguity is confirmed,
- one RED test locks the missing boundary,
- one minimal GREEN fix lands locally on the detail widget,
- focused verification is green,
- the ledger records whether the broader stage now looks closure-ready or still has one clear remaining gap.

---

## Current recommended next step

**Status:** delivered for the current v1 seam

The bounded `detail edit-vs-fork clarity` slice is now landed on the shared detail seam:
- edit path remains direct and compact,
- fork path now explicitly explains preservation intent,
- the edit-vs-fork decision boundary is clearer without introducing a new workflow model.

Current stage assessment:
- the `reuse / refine / fork operating path` stage now looks closure-ready for commit at the current bounded-v1 level.
