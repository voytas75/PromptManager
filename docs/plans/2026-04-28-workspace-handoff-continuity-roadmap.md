# Workspace Handoff Continuity Roadmap

Status: done
Owner: Wojtek / Prompt Manager Team
Updated: 2026-04-28

## Why this cycle now

The bounded `Prompt Actions Clarity` cycle is now closed in practice:
- `Show Description` empty-state guidance is verified as covered by existing behavior,
- duplicate / fork wording parity is locked by focused guards,
- recommendation / execution wording distinctions are locked by focused guards,
- execution-action locality wording is locked without changing runtime semantics.

The next smallest operator-facing seam near the core loop is the existing workspace handoff path:
- it sits directly between **inspect/reuse** and **refine**,
- it already has a shipped status/toast seam in `PromptActionsController.open_prompt_in_workspace()`,
- it already carries bounded validation-aware guidance, but that continuity should now be rechecked as its own execution boundary rather than left implicit.

This cycle stays deliberately narrow:
- reuse the existing `PromptActionsController.open_prompt_in_workspace()` seam,
- prefer status/toast continuity and wording confidence over behavior changes,
- avoid widening into execution automation, persistence, history redesign, or new workspace surfaces.

---

## Product fit

This cycle supports the SSOT core loop at:
- **Inspect** — preserve clarity about what evidence exists before handoff,
- **Reuse** — keep the operator aware of what to validate next,
- **Refine** — make the transition into workspace feel intentional rather than silent.

It remains subordinate to the product center:
- prompt assets first,
- workspace handoff as bounded support for reuse/refine continuity,
- no new workflow layer.

---

## Cycle theme

**Workspace handoff continuity on existing seams**

Goal:
Keep the existing workspace handoff easy to trust by locking the operator-facing continuity cues already present on the current seam.

Non-goals:
- no auto-run behavior,
- no new actions or buttons,
- no storage/model changes,
- no execution-history schema changes,
- no CLI/headless parity work unless a cue enters shared analytics truth.

---

## Candidate bounded slices

### Slice 1 — Generic workspace handoff continuity guard
**Status:** done (guard-only)

**Intent:** Reconfirm the default `Open in Workspace` handoff still seeds the prompt body without executing and still gives one explicit next-step validation hint.

**Seam used:**
- `gui/prompt_actions_controller.py` (`open_prompt_in_workspace()`)
- `tests/test_prompt_actions_controller.py`

**Outcome:**
- runtime already matched the intended behavior,
- toast remained unchanged,
- generic status hint remained explicit and bounded,
- the slice landed as a stronger guard test only.

**Locked by test:**
- workspace seeding still happens,
- no execution is triggered during handoff,
- the generic hint still points to validate/refine next,
- the generic hint stays distinct from the stale-validation variant.

### Slice 2 — Stale-validation handoff continuity parity guard
**Status:** done (guard-only)

**Intent:** Reconfirm the stale-validation variant stays stronger than the generic handoff hint while preserving the same non-executing workspace behavior.

**Seam used:**
- `gui/prompt_actions_controller.py` (`open_prompt_in_workspace()` / `_workspace_handoff_status_message()`)
- `tests/test_prompt_actions_controller.py`

**Outcome:**
- runtime already kept the stale-validation path stronger than the generic handoff hint,
- non-executing workspace seeding stayed unchanged,
- toast continuity stayed unchanged,
- the slice landed as a stronger parity guard only.

**Locked by test:**
- stale-validation handoff still seeds the workspace without execution,
- the stale hint still names the latest validation as stale,
- the stale hint stays distinct from the generic validate/refine wording,
- the stale variant remains the stronger operator cue.

### Slice 3 — Workspace handoff cue locality / action-local parity guard
**Status:** done (guard-only)

**Intent:** Reconfirm the handoff status remains action-local and does not drift into shared analytics or unrelated CLI surfaces.

**Seam used:**
- `gui/prompt_actions_controller.py` (`open_prompt_in_workspace()`)
- `tests/test_prompt_actions_controller.py`

**Outcome:**
- runtime already kept the handoff cue local to the open-in-workspace action,
- no execution side effect was introduced,
- no usage-logger side effects were triggered by the handoff cue path,
- the slice landed as an action-local guard only.

**Locked by test:**
- workspace seeding still happens,
- no execution is triggered,
- status/toast stay on the handoff action path,
- copy / execute / detect analytics remain untouched by open-in-workspace.

---

## First recommended slice

### Pick first: Slice 1 — Generic workspace handoff continuity guard

Why this first:
- smallest seam in the next execution boundary,
- already has a focused existing test path nearby,
- directly supports inspect → reuse → refine continuity,
- cheap to verify with prompt-actions tests,
- gives a clean probe before deciding whether later slices need runtime changes or only stronger guards.

---

## Execution rule for this cycle

For each slice:
1. read the exact runtime seam and nearest focused tests,
2. add one RED/probe test first,
3. verify whether it truly fails or is already covered,
4. implement the smallest wording-only/runtime-local fix only if needed,
5. run targeted + nearby smoke + quality gates,
6. update this roadmap immediately after green,
7. commit/push as one bounded slice.
