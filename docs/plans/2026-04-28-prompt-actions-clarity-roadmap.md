# Prompt Actions Clarity Roadmap

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-04-28

## Why this cycle now

The just-closed favorites clarity cycle improved retrieval intent on an existing filter/detail seam.
The next most valuable bounded area near the product center is the existing prompt actions context menu:
- it sits directly on the **reuse** step of the core loop,
- it already contains meaningful actions (`Duplicate Prompt`, `Fork Prompt`, `Similar Prompts`, `Execute Prompt`, `Show Description`),
- parts of it already received small tooltip clarity slices, but the seam still looks only partially explained and uneven in operator guidance.

This cycle keeps the work bounded:
- stay on the existing `PromptActionsController.show_context_menu()` seam unless a nearby dialog/text seam is strictly smaller,
- prefer tooltip/empty-state wording clarity over callback, routing, or persistence changes,
- avoid widening into execution flow, search ranking, editor workflow, or metadata generation.

---

## Product fit

This cycle supports the SSOT core loop at:
- **Inspect** — clarify what an action actually reveals or does,
- **Reuse** — make the next operator move easier to choose,
- **Refine** — preserve the distinction between duplicate/fork/recommendation paths.

It remains subordinate to the product center:
- prompt assets first,
- action clarity as bounded support for reuse,
- no new workflow layer.

---

## Cycle theme

**Prompt actions clarity on existing seams**

Goal:
Make the prompt context menu and the nearest adjacent action seams easier to trust at a glance without changing what the actions do.

Non-goals:
- no new actions,
- no execution semantics changes,
- no storage/model changes,
- no redesign of the menu or detail screen,
- no CLI/headless parity work unless a cue is promoted into shared analytics truth.

---

## Candidate bounded slices

### Slice 1 — Show Description empty-state clarity v1
**Status:** covered by existing behavior

**Intent:** Improve the empty-description message so the action explains the nearest useful next step instead of only reporting absence.

**Likely seam:**
- `gui/prompt_actions_controller.py` (`show_prompt_description()`)
- `tests/test_prompt_actions_controller.py`

**Good shape:**
- keep dialog title unchanged,
- change only the empty-state body copy,
- keep non-empty description behavior untouched.

**Verified reality:**
- the runtime seam already keeps the dialog title `No description available`,
- the empty-state body already says `The selected prompt does not have a description yet. Inspect the prompt body or add a short description for faster reuse.`,
- the nearest focused regression test already exists as `test_show_prompt_description_surfaces_guidance_when_description_is_missing`,
- no runtime/code change was needed for this slice.

**Verification target:**
- targeted offscreen pytest for the specific prompt-actions test,
- nearby prompt-actions smoke,
- `ruff check`, `ruff format --check`, `py_compile` on touched files.

### Slice 2 — Duplicate vs fork wording parity guard
**Status:** implemented

**Intent:** Reconfirm the distinction between duplication and fork lineage remains explicit and symmetric on the context-menu seam.

**Likely seam:**
- `gui/prompt_actions_controller.py` (`show_context_menu()`)
- `tests/test_prompt_actions_controller.py`

**Verified reality:**
- the runtime seam already keeps the duplicate/fork distinction explicit with distinct tooltips,
- `Duplicate Prompt` keeps `Create an editable copy of this prompt without fork lineage.`,
- `Fork Prompt` keeps `Create a fork linked to this prompt and open it for editing.`,
- the slice landed as a guard-strengthening test that asserts both tooltip strings together and proves they stay distinct on the same context-menu seam,
- no runtime/code change was needed for this slice.

**Verification target:**
- targeted offscreen pytest for the duplicate/fork parity test,
- nearby prompt-actions smoke,
- `ruff check`, `ruff format --check`, `py_compile` on touched files.

### Slice 3 — Similar Prompts recommendation cue continuity guard
**Status:** implemented

**Intent:** Reconfirm the recommendation path wording stays distinct from ordinary search and from direct execution.

**Likely seam:**
- `gui/prompt_actions_controller.py` (`show_context_menu()`)
- `tests/test_prompt_actions_controller.py`

**Verified reality:**
- the runtime seam already keeps `Similar Prompts` on explicit recommendation wording,
- `Similar Prompts` keeps `Show recommendation results for prompts similar to this one.`,
- the tooltip wording stays distinct from direct execution wording and avoids drifting into ordinary search language,
- the slice landed as a guard-strengthening test that asserts the recommendation tooltip remains explicit and distinct on the same context-menu seam,
- no runtime/code change was needed for this slice.

**Verification target:**
- targeted offscreen pytest for the Similar Prompts continuity test,
- nearby prompt-actions smoke,
- `ruff check`, `ruff format --check`, `py_compile` on touched files.

### Slice 4 — Execute / Execute as Context locality + meaning guard
**Status:** implemented

**Intent:** Reconfirm execution-action explanations stay bounded to the action seam and do not drift into broader workflow claims.

**Likely seam:**
- `gui/prompt_actions_controller.py` (`show_context_menu()`)
- `tests/test_prompt_actions_controller.py`

**Verified reality:**
- the runtime seam already keeps `Execute Prompt` and `Execute as Context…` on bounded action-local wording,
- `Execute Prompt` keeps `Run this prompt immediately using its stored text.`,
- `Execute as Context…` keeps `Run the stored prompt body as context for an ad-hoc task.` when enabled,
- the disabled `Execute as Context…` path already explains the missing stored prompt body without widening into broader workflow claims,
- the slice landed as a guard-strengthening test that asserts both execution tooltips stay distinct, local, and workflow-bounded on the same context-menu seam,
- no runtime/code change was needed for this slice.

**Verification target:**
- targeted offscreen pytest for the execute-action locality test,
- nearby prompt-actions smoke,
- `ruff check`, `ruff format --check`, `py_compile` on touched files.

---

## First recommended slice

### Pick first: Slice 1 — Show Description empty-state clarity v1

Why this first:
- smallest seam with clear operator-facing payoff,
- directly supports inspect/reuse readiness,
- bounded to one message path,
- cheap to verify with focused tests,
- avoids reopening already-clarified tooltip actions first.

---

## Execution rule for this cycle

For each slice:
1. read the exact runtime seam and nearest focused tests,
2. add one RED test first,
3. verify whether it truly fails or is already covered,
4. implement the smallest wording-only/runtime-local fix only if needed,
5. run targeted + nearby smoke + quality gates,
6. update this roadmap immediately after green,
7. commit/push as one bounded slice.
