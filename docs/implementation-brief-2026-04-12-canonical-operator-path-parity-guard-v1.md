# PromptManager — Implementation Brief

Date: 2026-04-12
Status: ready
Feature: Canonical Operator Path Parity Guard v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/canonical-usage-path-v1.md`
- `README.md`

## Goal

Lock one official operator path as a **live-aligned parity contract** between docs and the current UI:

**`Quick Capture` → `Promote Draft` → `Recent` / search → inspect → `Copy Prompt` or `Open in Workspace`**

This slice should act as a **guard against drift**, not as a new feature or broad UX/docs pass.

## Why this now

The repo already declares a canonical operator path in both:
- `README.md`
- `docs/canonical-usage-path-v1.md`

Recent bounded slices already improved adjacent seams substantially:
- search/list clarity,
- detail inspection cues,
- template preview clarity,
- quick reuse behavior.

At the same time:
- the Quick Capture real-input review did **not** justify a new cleanup rule yet,
- another micro-cue in search/detail/template would likely produce lower product value than protecting the path already declared as canonical.

The next strong move is therefore:

> make the declared canonical path stay true in docs and live UI over time

## Product intent

This slice strengthens the core loop by protecting the product’s main path across:
- capture,
- normalize,
- retrieve,
- inspect,
- reuse.

It should improve confidence that PromptManager’s documented front-door path is not drifting away from the actual app surface.

This is a **contract-locking slice**, not a workflow expansion slice.

## Scope

### In scope
- align one canonical operator path between:
  - `README.md`
  - `docs/canonical-usage-path-v1.md`
  - current shared UI labels and states
- add one deterministic regression guard for that path
- make only the smallest code/doc fixes required if a real parity mismatch is found
- anchor the contract specifically to the official path, not every alternate affordance in the app

### Out of scope
- broad README cleanup
- repo-wide terminology sweep
- onboarding/tutorial/wizard work
- full GUI click-through automation
- new product flows
- adjacent toolbar/detail refactors
- tooltip copy cleanup unless it is directly part of the contract
- changes to execution, analytics, chains, sharing, voice, or template subsystems

## Recommended UX posture

Prefer **parity guard** over richer workflow mechanics.

Suggested v1 posture:
- keep the official path small and explicit,
- keep labels stable only where they define the canonical path,
- guard the path with one focused test rather than a broad GUI automation harness,
- avoid freezing every nearby wording detail.

Default recommendation:
- **lock the path, not the whole surface**

## Likely implementation seams

### Docs
- `README.md`
- `docs/canonical-usage-path-v1.md`

### UI/code seams most likely to matter
- `gui/widgets/prompt_toolbar.py`
  - `Quick Capture`
  - `Recent`
- `gui/widgets/prompt_detail_widget.py`
  - `Promote Draft`
  - `Copy Prompt`
  - `Open in Workspace`
- possibly `gui/main_window_handlers.py` only if recent reopen or selection handoff needs a minimal alignment fix

### Test seams
Prefer extending current focused tests where possible:
- `tests/test_prompt_toolbar.py`
- `tests/test_recent_prompts.py`
- `tests/test_prompt_detail_widget.py`

Add one small dedicated contract/parity test, likely something like:
- `tests/test_canonical_operator_path_parity.py`

That test should remain deterministic and avoid brittle end-to-end GUI scripting.

## Recommended test posture

Do **not** build a large GUI automation path for this slice.

Preferred v1 guard:
- assert that docs declare the same canonical path,
- assert that the shared UI still exposes the corresponding labels/seams,
- assert that detail-state gating still matches the official operator path,
- keep the test bounded to contract presence and parity rather than click-by-click orchestration.

## Happy-path contract

### Path A: capture to reusable asset
1. Operator uses `Quick Capture` from the toolbar.
2. The captured item exists as a draft prompt in the existing flow.
3. The operator uses `Promote Draft` from the detail view.
4. The asset becomes a reusable prompt record in the catalog.

### Path B: reopen and inspect
1. Operator reopens the prompt from `Recent` or search.
2. The prompt is visible in the normal inspect/detail flow.
3. The operator confirms fit using the existing detail surface.

### Path C: reuse
1. Operator uses `Copy Prompt` when the stored prompt body is the desired output.
2. Or uses `Open in Workspace` when the current reusable text should be handed off into the workspace path.
3. No new execution behavior or alternate workflow is introduced.

That is enough for v1.

## Acceptance checks

1. `README.md` and `docs/canonical-usage-path-v1.md` both state the same canonical path:
   - `Quick Capture` → `Promote Draft` → `Recent` / search → inspect → `Copy Prompt` or `Open in Workspace`
2. Toolbar still exposes:
   - `Quick Capture`
   - `Recent`
3. Detail view still exposes, in the relevant states:
   - `Promote Draft` for drafts
   - `Copy Prompt` when prompt body exists
   - `Open in Workspace` when reusable text exists
4. Recent reopen behavior remains deterministic in the current handler seam.
5. One deterministic regression test fails if the canonical path drifts between docs and the live shared UI contract.
6. No broader UX redesign, repo-wide cleanup, or new operator path is introduced.

## Rollback

Rollback should be one isolated patch:
- revert the narrow docs changes,
- revert the dedicated parity/contract test,
- revert any minimal UI alignment fix added only for this slice,
- leave adjacent prompt-list/detail/template/reuse behavior untouched.

## Anti-goals

- do not turn this into a broad docs pass
- do not freeze all wording across the repo
- do not include every alternate entry point as part of the contract
- do not build brittle full GUI end-to-end automation
- do not use this slice to reopen unrelated polish work
- do not bundle a second product slice into the same patch

## Notes for implementation

- Keep the slice boring.
- The contract should protect the official front door, not document every possible route through the app.
- If implementation starts expanding into tutorials, walkthroughs, or generalized terminology cleanup, the slice is drifting.
- If a mismatch is found, prefer the smallest patch that restores parity rather than redesigning the seam.
