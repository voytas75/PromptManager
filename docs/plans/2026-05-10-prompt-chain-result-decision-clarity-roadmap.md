# PromptManager Prompt-Chain Result Decision Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce post-run hesitation on the existing prompt-chain result surface so the operator can tell what to consume first when a chain returns both a final output and a final summary, without expanding the chain engine, storage model, or product scope.

**Architecture:** This roadmap stays on one existing seam: the prompt-chain result presentation surface in `gui/dialogs/prompt_chains.py`. It treats `final_output_text` as the primary consumable artifact, keeps `final_summary_text` as secondary supporting interpretation, and improves operator judgment through compact wording only. The slice is GUI-local unless a later parity check proves otherwise.

**Tech Stack:** Python 3.13, PySide6 GUI seam, pytest, Ruff, prompt-chain dialog tests, existing product docs under `docs/`.

Direction note: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Canonical product SSOT: `docs/product-ssot.md`
Most recently delivered bounded ledger before this slice: `docs/plans/2026-05-10-detail-refine-fork-action-clarity-roadmap.md`

---

## Why this roadmap exists

Confirmed from current repo state:
- `docs/product-ssot.md` keeps prompt-chain work subordinate to the prompt-asset product center and explicitly discourages heavy prompt-chain expansion.
- `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` keeps Candidate 4 pending as bounded prompt-chain ergonomics only:
  - no engine expansion,
  - no chain-first product shift,
  - only clarity / inspectability / validation / consumption improvements.
- `gui/dialogs/prompt_chains.py` already renders separate `Final output` and `Final summary` blocks and already exposes separate copy actions for output vs summary.
- `tests/test_prompt_chain_dialog.py` previously proved that the GUI shows the summary block but did not yet prove that the summary is framed as supporting context rather than an equal peer to the final output.
- `tests/test_prompt_chain_cli.py` already keeps explicit `Final output:` vs `Final summary:` semantics and selective output modes, so the current hesitation seam is GUI-local unless a later parity check proves otherwise.

So the next useful move is:

**bounded prompt-chain result consumption clarity on the existing GUI result seam**

---

## Scope guardrails

This roadmap may improve only:
- one bounded hesitation seam on the existing prompt-chain result surface,
- compact wording that makes the primary result easier to consume,
- focused tests proving primary-vs-supporting framing,
- bounded non-regression verification for the touched GUI seam.

This roadmap must not introduce:
- chain-engine expansion,
- new run modes,
- new persistence/history behavior,
- a prompt-chain-first product story,
- new automation surfaces,
- docs claims that overstate persistence or workflow scope.

---

## Confirmed baseline

Treat these as already delivered unless focused regression proves otherwise:
- prompt-chain GUI already separates input, step outputs, final output, and summary,
- prompt-chain GUI already supports copying final output and final summary independently,
- prompt-chain CLI already distinguishes `Final output:` from `Final summary:` and offers selective output modes,
- earlier prompt-chain slices already delivered recent-history alignment, result semantics separation, and calmer output labeling,
- detail/refine-fork work is the most recently delivered bounded slice before this one.

Do not re-plan those as missing features.

---

## Main product question for this roadmap

> when a prompt chain returns both a final output and a final summary, how do we make the GUI tell the operator what to consume first without widening prompt-chain scope or rewriting shared execution semantics?

Preferred answer shape:
- one bounded result-consumption seam,
- one clearer primary-vs-supporting framing,
- no workflow expansion,
- all on the existing prompt-chain result surface.

---

## Stage A — Prompt-chain result decision clarity

### Task 1: Audit the current result-consumption hesitation seam

**Status:** completed

**Objective:** Confirm the smallest still-ambiguous operator decision on the existing prompt-chain result surface before implementation starts.

**Chosen bounded hesitation seam (confirmed):**
- The GUI already renders `Final output` before `Final summary`, so the ordering is not the missing behavior.
- The GUI already exposes separate copy actions for output and summary, so the capability model already distinguishes the two artifacts.
- The remaining bounded ambiguity is therefore not missing output/summary support; it is missing **presentation clarity** that tells the operator the summary is supporting context rather than an equal peer to the final output.
- Existing dialog coverage previously asserted only that the summary block appears and that older wording like `Final chain result` stays absent.
- No focused GUI test previously asserted that the summary reads as supporting interpretation.
- The selected v1 direction is therefore **one GUI-local wording/cue improvement for the summary block**, without changing backend payloads, storage, CLI modes, or workflow scope.

**Audit notes (confirmed from code/tests):**
- `gui/dialogs/prompt_chains.py` renders `Final output` and `Final summary` as separate result blocks, but both use neutral section titles.
- `gui/dialogs/prompt_chains.py` keeps `_last_final_output_text` and `_last_final_summary_text` separately and exposes separate copy actions.
- `tests/test_prompt_chain_dialog.py` previously covered summary presence and stale-result replacement, but not primary-vs-supporting framing.
- `tests/test_prompt_chain_cli.py` search confirms the CLI already uses explicit `Final output:` / `Final summary:` wording and should stay untouched unless parity later proves otherwise.

**Files:**
- Inspect: `gui/dialogs/prompt_chains.py`
- Inspect: `tests/test_prompt_chain_dialog.py`
- Reference only if needed: `tests/test_prompt_chain_cli.py`, `tests/test_prompt_chain_backend.py`
- Maintain: `docs/plans/2026-05-10-prompt-chain-result-decision-clarity-roadmap.md`

**Implemented / audited:**
- Confirmed that the smallest useful seam is GUI-local wording only.
- Chosen fix direction: reframe the summary block as `Supporting summary` while keeping `Final output` unchanged and primary.

**Verified:**
- inspected `gui/dialogs/prompt_chains.py` final output / final summary render path,
- inspected copy/save affordances for output vs summary,
- inspected dialog tests around summary rendering,
- searched CLI tests for current output/summary contract.

---

### Task 2: Add one RED dialog test for supporting-summary framing

**Status:** completed

**Objective:** Prove the missing GUI cue before changing implementation.

**Files:**
- Modify: `tests/test_prompt_chain_dialog.py`

**Chosen RED contract:**
- when both result blocks exist,
- the result surface should still include `Final output`,
- and the summary should render as `Supporting summary`.

**Implemented:**
- added `test_prompt_chain_dialog_marks_final_summary_as_supporting_context`.

**Verified:**
- `pytest tests/test_prompt_chain_dialog.py::test_prompt_chain_dialog_marks_final_summary_as_supporting_context -q`
- result: `FAIL`
- failure confirmed current text still used `Final summary` rather than the supporting-context wording.

---

### Task 3: GREEN minimal GUI-local wording fix

**Status:** completed

**Objective:** Make the summary read as secondary supporting context without widening prompt-chain scope.

**Files:**
- Modify: `gui/dialogs/prompt_chains.py`
- Modify: `tests/test_prompt_chain_dialog.py`

**Implemented:**
- changed the GUI result block label from `Final summary` to `Supporting summary` in both plain-text and rich-text render paths,
- kept `Final output` unchanged and primary,
- updated focused dialog tests and nearby non-regression expectations that previously asserted the old label.

**Verified:**
- `pytest tests/test_prompt_chain_dialog.py::test_prompt_chain_dialog_marks_final_summary_as_supporting_context -q` -> `1 passed`
- `pytest tests/test_prompt_chain_dialog.py -q` -> `46 passed`
- `ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py` -> `All checks passed!`

**Parity note:**
- no CLI/backend changes were required because the shipped fix stayed on a GUI-local presentation seam.

---

### Task 4: Sync the execution ledger and pointer-chain after the bounded slice lands

**Status:** completed

**Objective:** Keep next-cycle docs unambiguous after opening and landing this bounded C4.1 slice.

**Files:**
- Modify: this file
- Modify: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
- Modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Implemented:**
- opened this fresh execution ledger for the C4.1 slice,
- updated the direction note so Candidate 4 became the active current slice for this bounded roadmap,
- updated the umbrella next-cycle pointer to this ledger,
- preserved `docs/product-ssot.md` unchanged because product truth did not move.

**Verified:**
- pointer-chain re-read across the three planning files,
- targeted docs search for active execution-ledger wording,
- confirmed one active bounded ledger for the current next-cycle slice.

---

## Current recommended next slice

This bounded v1 seam is delivered.

Do not treat this ledger as an open implementation pointer anymore.
The next bounded slice must be selected in a later revision of the direction note and/or a fresh execution ledger.

---

## Closure / successor rule

After this delivery:
- do not reopen the shipped `Supporting summary` wording seam itself,
- choose a successor only if it remains on one existing prompt-chain ergonomics seam such as validation clarity or inspectability,
- do not let prompt-chain work displace the prompt-asset product center.

---

## Verification pack for this slice

Executed / expected commands:
```bash
pytest tests/test_prompt_chain_dialog.py::test_prompt_chain_dialog_marks_final_summary_as_supporting_context -q
pytest tests/test_prompt_chain_dialog.py -q
ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
```

Conditional parity commands were intentionally skipped because the delivered fix stayed GUI-local:
```bash
pytest tests/test_prompt_chain_cli.py -q
pytest tests/test_prompt_chain_backend.py -q
```
