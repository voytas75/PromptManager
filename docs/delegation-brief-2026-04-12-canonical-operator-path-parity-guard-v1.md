# PromptManager — Delegation Brief

Date: 2026-04-12
Status: ready-for-delegation
Target: dev / Codex
Feature: Canonical Operator Path Parity Guard v1
Primary brief:
- `docs/implementation-brief-2026-04-12-canonical-operator-path-parity-guard-v1.md`

## Mission

Implement one **small parity-guard slice** for the declared canonical operator path:

**`Quick Capture` → `Promote Draft` → `Recent` / search → inspect → `Copy Prompt` or `Open in Workspace`**

This is **not** a new feature.
It is a bounded alignment pass to make sure:
- docs describe the same path,
- live UI still exposes the corresponding contract seams,
- one deterministic regression test catches future drift.

## Required posture

Keep this slice:
- small,
- boring,
- implementation-minded,
- local in effect,
- free of adjacent cleanup.

If you find a mismatch, prefer the **smallest patch that restores parity**.
Do not redesign the surrounding UX.

## Source anchors

Read first:
- `docs/implementation-brief-2026-04-12-canonical-operator-path-parity-guard-v1.md`
- `docs/canonical-usage-path-v1.md`
- `README.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

Likely code seams:
- `gui/widgets/prompt_toolbar.py`
- `gui/widgets/prompt_detail_widget.py`
- `gui/main_window_handlers.py` only if needed

Likely existing tests to extend:
- `tests/test_prompt_toolbar.py`
- `tests/test_recent_prompts.py`
- `tests/test_prompt_detail_widget.py`

Possible new focused test:
- `tests/test_canonical_operator_path_parity.py`

## Deliverable

Ship exactly one bounded patch that does all of the following:
1. confirms or minimally aligns the canonical operator path between docs and live UI
2. adds one deterministic parity/contract regression check
3. updates docs only where needed for strict parity

## Do now

### 1. Verify the path contract
Check that these contract elements are really present and named consistently:
- toolbar:
  - `Quick Capture`
  - `Recent`
- detail flow:
  - `Promote Draft` for drafts
  - `Copy Prompt` when prompt body exists
  - `Open in Workspace` when reusable text exists
- reopen path:
  - `Recent` or search leads back into the existing inspect/detail flow

### 2. Patch only real mismatches
If docs and UI already match, keep code changes at zero or near-zero.
If they do not match, apply the smallest viable patch.

### 3. Add one deterministic guard
Preferred guard shape:
- assert the canonical path wording is present and aligned in docs,
- assert the corresponding shared UI labels/seams still exist,
- avoid brittle click-by-click GUI automation.

## Acceptance checks

1. `README.md` and `docs/canonical-usage-path-v1.md` state the same canonical operator path.
2. Toolbar still exposes:
   - `Quick Capture`
   - `Recent`
3. Detail surface still exposes in relevant states:
   - `Promote Draft`
   - `Copy Prompt`
   - `Open in Workspace`
4. Recent reopen remains deterministic in the current seam.
5. One deterministic regression test fails if docs and live UI contract drift apart.
6. No new workflow, no broad docs sweep, no unrelated UI cleanup is introduced.

## Validation

Run focused validation only.
Prefer the narrowest reasonable test set that proves the slice:
- the touched focused tests,
- the new parity/contract test if added.

If no new dedicated test file is needed because the guard fits better into existing tests, that is acceptable.

## Required final report

Return:
1. what changed,
2. exact files changed,
3. validation run and results,
4. whether any mismatch was found,
5. whether the slice stayed bounded.

## Anti-goals

- do not turn this into a broad docs cleanup
- do not rename unrelated labels across the repo
- do not freeze tooltip wording unless it is truly part of the contract
- do not build a large GUI automation harness
- do not refactor toolbar/detail architecture
- do not mix in another product slice

## Rollback

Rollback should be one isolated revert of:
- parity docs edits,
- the dedicated regression guard,
- any minimal UI alignment patch added only for this slice.
