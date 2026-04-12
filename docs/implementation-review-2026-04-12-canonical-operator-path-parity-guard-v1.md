# PromptManager — Implementation Review

Date: 2026-04-12
Target: `Canonical Operator Path Parity Guard v1`
Expected source: `docs/implementation-brief-2026-04-12-canonical-operator-path-parity-guard-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. The canonical operator path was already in parity across README, the canonical usage doc, and the current UI seams, so the implementation correctly stayed minimal and shipped only one deterministic regression guard instead of widening into docs cleanup or UI changes.

## What matches

### 1. The slice stayed in the intended parity-guard posture
The implementation did not introduce a new feature or workflow.
It added one dedicated contract test that locks the declared short path to the existing live seams.

That matches the brief's intended posture exactly:
- protect the official path,
- patch only if mismatch exists,
- avoid broader cleanup.

### 2. Docs and live UI are now guarded together
The added test verifies that both:
- `README.md`
- `docs/canonical-usage-path-v1.md`

still contain the same canonical path string:

- `Quick Capture` → `Promote Draft` → `Recent` / search → inspect → `Copy Prompt` or `Open in Workspace`

That is the core contract the brief asked to lock down.

### 3. The right live seams are covered
The regression guard checks the expected shared seams directly:
- toolbar labels:
  - `Quick Capture`
  - `Recent`
- detail states:
  - `Promote Draft` visible/enabled for drafts
  - `Copy Prompt` enabled when prompt body exists
  - `Open in Workspace` enabled when reusable text exists, including the description-only seam
- recent ordering:
  - deterministic ordering through the existing `recent_prompts(...)` seam

That is proportionate and matches the brief better than a brittle click-through GUI test would.

### 4. Focused validation passed
Focused validation completed successfully:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_canonical_operator_path_parity.py tests/test_prompt_toolbar.py tests/test_recent_prompts.py tests/test_prompt_detail_widget.py`
- result: `21 passed`

That is strong evidence for this slice.

## What is missing

Nothing material relative to the brief.

The slice did not need docs edits or UI alignment patches because no real mismatch was found.
That is an acceptable and correct outcome for a parity-guard implementation.

## What drifted / widened

No meaningful scope drift is visible.

The implementation avoided:
- broad README cleanup,
- repo-wide terminology work,
- tooltip freeze/polish,
- full GUI automation,
- toolbar/detail refactors,
- a second bundled slice.

That restraint is part of why this result is good.

## What is unverified

### 1. Full click-through operator flow in a live interactive session
This review confirms contract presence and seam parity, but it does not run a human-like end-to-end GUI interaction sequence through capture, promote, reopen, and reuse.

That is acceptable here because the brief explicitly preferred a deterministic parity guard over brittle GUI automation.

### 2. Alternate entry points outside the canonical path
This review does not verify every other way to reach promote/reuse actions in the app.
It verifies only the declared official path.

That is also acceptable because the brief intentionally scoped the contract to the canonical path only.

## Recommended next action

Treat `Canonical Operator Path Parity Guard v1` as delivered.

Do not widen it into a larger docs or UX pass.
If a future change alters the official front-door path, update:
- `README.md`
- `docs/canonical-usage-path-v1.md`
- `tests/test_canonical_operator_path_parity.py`

in the same slice so the parity guard remains honest.

## Sources reviewed

- `docs/implementation-brief-2026-04-12-canonical-operator-path-parity-guard-v1.md`
- `README.md`
- `docs/canonical-usage-path-v1.md`
- `tests/test_canonical_operator_path_parity.py`
- focused validation result:
  - `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_canonical_operator_path_parity.py tests/test_prompt_toolbar.py tests/test_recent_prompts.py tests/test_prompt_detail_widget.py` → `21 passed`
