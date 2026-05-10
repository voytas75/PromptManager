# PromptManager Prompt-List Result Expectation Clarity Roadmap

- **Date:** 2026-05-10
- **Owner:** Hermes
- **Status:** proposed active bounded slice
- **Depends on:**
  - `docs/product-ssot.md`
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-05-10-search-result-action-clarity-roadmap.md`
- **Target seam:**
  - `gui/prompt_list_model.py`
  - `tests/test_prompt_list_model.py`
  - maybe `gui/prompt_list_delegate.py`
  - maybe `tests/test_prompt_list_presenter.py`
- **Out of scope:** ranking changes, retrieval-model redesign, detail-view decision semantics, persistence, new retrieval layers, global workflow expansion.

---

## Why this slice exists

The previous bounded retrieval slice already landed the v1 family:
- `Matched in title` -> `Likely reusable as-is`
- `Matched in source` -> `Inspect before reuse`
- `Matched in scenario` -> `Inspect before reuse`

That was the right first cut, but it still leaves one smaller follow-up seam on the prompt-list surface:
- the list now tells the operator whether a row looks more reuse-ready or inspect-first,
- but the title-match wording still reads slightly weaker than the product intent of a fast reuse candidate,
- and the current list-local handoff contract does not yet prove a stronger expectation distinction between title matches and non-title matches.

This slice stays bounded by refining only the row-level expectation wording/contract on the prompt-list seam.

---

## Product question

> Can the prompt list make the difference between a likely fast-reuse candidate and an inspect-first candidate more explicit, without copying detail-view `Decision` / `Decision basis` semantics?

Preferred answer shape:
- keep the same three existing match-reason families,
- refine only one compact title-match expectation cue if needed,
- preserve inspect-first wording for non-title matches unless the audit proves it should change too,
- keep the result list-local and GUI-local.

---

## Task 1 — Verify the current expectation-gap precisely
**Status:** completed

**Objective:** Confirm whether the remaining gap is real and small enough for one wording/contract slice.

**Confirmed gap:**
- the current title-match cue `Likely reusable as-is` is directionally correct, but it still reads weaker than the intended fast-reuse expectation and does not create as sharp a contrast as it could against the inspect-first non-title cue family.

**Audit notes:**
- `gui/prompt_list_model.py` currently exposes only two list-local expectation families:
  - `Matched in title` -> `Likely reusable as-is`
  - `Matched in source` / `Matched in scenario` -> `Inspect before reuse`
- `tests/test_prompt_list_model.py` locks those exact strings, so the remaining gap is now strictly a wording/contract refinement seam rather than a missing cue family.
- The non-title cue family is still product-sound and already aligned with presenter-level global inspect framing.
- The smallest meaningful follow-up is therefore **title-match expectation clarity only**, not a broader rewrite of all list-side cues.

**Chosen slice rule:**
- keep `Inspect before reuse` unchanged for non-title matches,
- refine only the title-match cue to read more like an immediate reuse expectation,
- keep the change on `gui/prompt_list_model.py` unless Task 2 proves otherwise.

---

## Task 2 — Add one RED test for the chosen expectation contract
**Status:** completed

**Objective:** Freeze the smallest missing operator-facing expectation contract before implementation.

**Implemented:**
- tightened the existing title-match handoff test in `tests/test_prompt_list_model.py`,
- changed the expected title-match cue from `Likely reusable as-is` to `Ready to reuse`,
- kept non-title source/scenario expectations unchanged so the slice stayed focused on title-match expectation clarity only.

**Verified:**
- `pytest tests/test_prompt_list_model.py -q` -> RED before implementation (`1 failed` on the old title-match cue wording).

---

## Task 3 — Implement the smallest GREEN fix on the existing prompt-list seam
**Status:** completed

**Objective:** Land the smallest bounded list-side change that satisfies Task 2.

**Implemented:**
- updated `gui/prompt_list_model.py` on the existing `HandoffCueRole` seam only,
- refined the title-match cue:
  - `Matched in title` -> `Ready to reuse`
- preserved the non-title cue family unchanged:
  - `Matched in source` / `Matched in scenario` -> `Inspect before reuse`
- kept the slice model-local; no delegate, presenter, ranking, or preview-selection behavior changed.

---

## Task 4 — Verify and close the ledger
**Status:** completed

**Objective:** Prove the slice landed cleanly and keep pointer docs unambiguous.

**Verified:**
- `pytest tests/test_prompt_list_model.py -q` -> `21 passed`
- `ruff check gui/prompt_list_model.py tests/test_prompt_list_model.py` -> `All checks passed!`
- active pointer chain now points at this ledger:
  - `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
  - `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Docs note:**
- no `README.md` or `docs/product-ssot.md` update needed because this slice only refined one existing list-local expectation cue.

---

## Definition of done

This slice is done when:
- one real remaining expectation-gap is confirmed,
- one RED test proves the missing list-side contract,
- one minimal GREEN change lands on the prompt-list seam,
- focused verification is green,
- the ledger records the delivered bounded result without reopening broader retrieval work.

---

## Current recommended next step

**Status:** delivered for the current v1 seam

The bounded `prompt-list result expectation clarity` slice is now landed on the prompt-list seam:
- `Matched in title` now maps to `Ready to reuse`
- `Matched in source` / `Matched in scenario` still map to `Inspect before reuse`

This closes the chosen title-vs-non-title expectation refinement without reopening ranking, preview selection, or detail-view semantics.
