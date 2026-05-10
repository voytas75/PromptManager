# PromptManager Detail Copy-vs-Workspace Action Clarity Roadmap

- **Date:** 2026-05-10
- **Owner:** Hermes
- **Status:** proposed active bounded slice
- **Depends on:**
  - `docs/product-ssot.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-05-10-prompt-list-result-expectation-clarity-roadmap.md`
- **Target seam:**
  - `gui/widgets/prompt_detail_widget.py`
  - `tests/test_prompt_detail_widget.py`
- **Out of scope:** controller rewrites, workspace workflow expansion, new execution semantics, ranking/search changes, new action surfaces.

---

## Why this slice exists

The prompt list now better distinguishes:
- fast reuse candidates (`Ready to reuse`), and
- inspect-first candidates (`Inspect before reuse`).

That makes the next operator hesitation seam more obvious on the detail surface:
- when the prompt is already selected,
- and both `Copy Prompt` and `Open in Workspace` are available,
- the UI still explains each button separately through tooltips,
- but it does not yet prove one compact action-level expectation contract for the copy-vs-workspace choice itself.

This slice stays bounded by clarifying only that existing detail-side choice.

---

## Product question

> When both `Copy Prompt` and `Open in Workspace` are available, can the detail surface make the safer immediate move clearer without adding a new workflow layer?

Preferred answer shape:
- keep both existing actions,
- keep tooltip-level semantics intact unless the audit proves they are wrong,
- add or refine only one compact detail-side contract,
- avoid duplicating full decision-package semantics into action tooltips.

---

## Task 1 — Verify the current copy-vs-workspace gap
**Status:** completed

**Objective:** Confirm the smallest real hesitation seam between the existing quick-reuse actions.

**Confirmed gap:**
- when both actions are enabled for a prompt with a stored body, the detail widget explains `Copy Prompt` and `Open in Workspace` separately through tooltips, but it does not yet state the safer immediate expectation that copying is the direct reuse path while workspace remains the deeper follow-up path.

**Audit notes:**
- `gui/widgets/prompt_detail_widget.py` already enables both actions independently and keeps their existing semantics bounded:
  - `Copy Prompt` -> `Copy the stored prompt body.`
  - `Open in Workspace` -> `Open the stored prompt body in the workspace without running it.`
- `tests/test_prompt_detail_widget.py` already locks enablement, button labels, tooltips, and signal emission for the quick-reuse pair.
- Existing workspace handoff cues cover validation / inspect / baseline / edit / fork paths, but there is no compact action-level cue for the simpler direct-reuse state when both quick-reuse actions are available.
- The smallest bounded follow-up is therefore one widget-local action cue for the direct reuse path, not a rewrite of button behavior or a controller change.

---

## Task 2 — Add one RED test for the chosen action contract
**Status:** completed

**Objective:** Freeze one missing detail-side action expectation before implementation.

**Implemented:**
- added one focused test in `tests/test_prompt_detail_widget.py` for the direct-reuse state where:
  - a prompt has a stored body,
  - both `Copy Prompt` and `Open in Workspace` are enabled,
  - the decision is `Reuse as-is`,
  - and the detail seam must expose `Next step: Copy Prompt for direct reuse.`
- the first RED run failed because the widget kept the workspace handoff cue hidden in this simpler direct-reuse path.

---

## Task 3 — Implement the smallest GREEN fix on the detail seam
**Status:** completed

**Objective:** Land the smallest bounded copy-vs-workspace clarity improvement that satisfies Task 2.

**Implemented:**
- kept both existing actions and tooltip semantics unchanged,
- added one direct-reuse branch in `_resolve_workspace_handoff_cue(...)` so the widget now shows:
  - `Next step: Copy Prompt for direct reuse.`
- made the branch work both when the next-action label explicitly carries `Reuse as-is` and when that text is hidden as a duplicate of the decision summary.

**Why this is minimal:**
- no controller changes,
- no action removal,
- no workflow expansion,
- only one bounded handoff cue on the existing detail seam.

---

## Task 4 — Verify and close the ledger
**Status:** completed

**Objective:** Prove the slice landed cleanly and keep pointer docs unambiguous.

**Verified:**
- `pytest tests/test_prompt_detail_widget.py -q` -> `42 passed`
- `ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`

**Pointer status:**
- active next-cycle ledger pointer already targets this file in:
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`

---

## Definition of done

This slice is done when:
- one real copy-vs-workspace hesitation seam is confirmed,
- one RED test proves the missing contract,
- one minimal GREEN fix lands on the detail widget,
- focused verification is green,
- the ledger records the delivered bounded result.

---

## Current recommended next step

**Status:** delivered for the current v1 seam

The bounded `detail copy-vs-workspace action clarity` slice is now landed on the shared detail seam:
- direct-reuse prompts with a stored body still expose both actions,
- tooltips remain unchanged and action-specific,
- the widget now adds one compact handoff cue:
  - `Next step: Copy Prompt for direct reuse.`

This closes the chosen hesitation seam without broadening workflow semantics.
