# PromptManager Asset-to-Run-to-Refine Coherence v3 Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć kolejny czysty execution cycle po domknięciu evaluation/governance track i dalej wzmacniać asset-to-run-to-refine loop bez naruszania asset-first SSOT.

**Architecture:** Ten roadmap traktuje wcześniejsze plany z 2026-04-25 i 2026-04-26 jako zamknięte execution boundaries. Nowy cykl nie zmienia produktu na dashboard-first ani workflow-first. Zamiast tego doprecyzowuje ciągłość między inspect/detail, ostatnim run evidence, reuse/refine decision oraz wejściem do workspace, używając istniejących seams w `WorkspaceHistoryController`, `PromptDetailWidget`, `PromptActionsController`, shared analytics i powiązanych testach.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, existing PromptManager execution history, inspect/detail surfaces, workspace handoff flow, CLI/headless shared analytics, docs under `docs/`.

---

## Why this roadmap exists

Live repo review confirms:
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` is fully implemented in scope,
- `docs/plans/2026-04-26-next-cycle-roadmap.md` is fully implemented in scope,
- `docs/plans/2026-04-26-next-cycle-closure-and-next-plan.md` already closed the evaluation/governance cycle and partially consumed the next-track probes,
- `docs/product-ssot.md` remains aligned with the shipped asset-first / trustworthy-runs posture,
- git worktree is currently clean on `master` at `7330dbe`.

So the next useful move is to start a fresh ledger instead of extending a transition document that is already mostly exhausted.

Short version:

**assets first -> trustworthy runs -> bounded evaluation/governance -> asset-to-run-to-refine coherence v3**

---

## Confirmed baseline before this cycle

Assume already delivered:
- deterministic settings, routing, and trust diagnostics,
- prompt-linked structured runs and bounded comparison evidence,
- inspect/detail decision, next-action, provenance, and freshness cues,
- stale-validation-aware workspace handoff hint,
- parity across shared/template detail surfaces,
- CLI/headless parity only for shared semantics,
- limited-evidence and clear/reset parity guards for the main inspect/detail path.

This cycle should not repeat those foundations.
It should improve how the operator understands the next bounded step after seeing run evidence.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager umie już pokazać run evidence i bounded guidance, to jak zrobić, żeby przejście z evidence do kolejnego sensownego kroku było jeszcze bardziej czytelne, ale nadal lekkie?

Nowy cykl wzmacnia trzy rzeczy:

1. **result-to-next-step continuity** — po zobaczeniu run evidence operator szybciej rozumie, czy ma reuse, refine, validate, czy wrócić do baseline;
2. **workspace handoff clarity** — wejście do workspace ma lepiej odzwierciedlać aktualny stan evidence bez uruchamiania nowego workflow;
3. **shared-semantics discipline** — GUI może rosnąć tylko wtedy, gdy CLI/headless albo świadomie dostaje ten sam shared meaning, albo pozostaje bez zmian z jawnym uzasadnieniem.

---

## Constraints

Do not add:
- a new dashboard,
- a new review queue,
- a new orchestration canvas,
- a new persistence model only for guidance,
- background schedulers or auto-runs,
- a CLI-only or workspace-only shadow decision model,
- broad analytics-first UX.

Reuse first:
- `gui/workspace_history_controller.py`
- `gui/widgets/prompt_detail_widget.py`
- `gui/prompt_actions_controller.py`
- `core/history_tracker.py`
- `cli/commands.py`
- existing tests around workspace history, detail widgets, prompt actions, and main entry

---

## Roadmap stages for this cycle

### Stage A — Fresh vs stale follow-through clarity

Cel: doprecyzować, jak inspect/detail i workspace sygnalizują kolejny krok po świeżym albo starym evidence.

Obszary:
- fresh-run next-step wording,
- stale-vs-fresh symmetry guards,
- bounded validation/reuse/refine handoff wording.

Constraint:
- no new labels unless the current seam cannot express the cue,
- prefer wording refinements and parity guards over new state.

### Stage B — Reuse vs refine legibility after evidence

Cel: zmniejszyć niepewność po obejrzeniu wyniku, gdy prompt wygląda „usable”, ale operator nadal powinien rozumieć, czy bezpieczniej reuse czy refine.

Obszary:
- compact reuse/refine wording alignment,
- bounded evidence-aware fallback phrasing,
- no new branching workflow.

Constraint:
- keep `Decision` conservative,
- use `Recommended next action` first before inventing new surfaces.

### Stage C — Workspace validation loop coherence

Cel: poprawić ciągłość między inspect/detail a `Open in Workspace`, bez zmiany semantyki tej akcji.

Obszary:
- status hint clarity,
- result-to-workspace continuity,
- symmetry between stale and fresh evidence cases.

Constraint:
- workspace handoff stays non-executing,
- no separate validation panel.

### Stage D — Shared-semantics parity discipline

Cel: utrzymać jasną regułę: jeśli bounded semantic staje się częścią shared analytics truth, musi mieć parity; jeśli jest action-local, CLI pozostaje unchanged by design.

Obszary:
- `history-analytics` parity only for shared fields,
- explicit no-op documentation for workspace-local semantics,
- regression guards for wording drift.

Constraint:
- no headless feature growth for its own sake.

---

## Recommended execution order

1. fresh-vs-stale next-step symmetry
2. reuse-vs-refine wording after evidence
3. workspace handoff coherence for both fresh and stale paths
4. shared-semantics parity or deliberate no-op
5. test-only guard slices that lock the operator path

---

## Candidate bounded slices

### Slice 1 — Fresh-validation next-step cue

**Status:** covered by existing behavior

**Intent:** Uzupełnić istniejący stale-validation path o symetryczny, bounded guidance dla sytuacji, gdy latest validation jest świeże i evidence nie wymaga silniejszego compare/baseline branch.

**Good shape:**
- reuse existing `Recommended next action` seam,
- no new persistence,
- no new decision state,
- only one compact wording refinement.

**Examples:**
- `Reuse current prompt`
- `Reuse current version`
- or another compact wording if tests prove the current copy is unclear.

**Probe result:**
- Current runtime already keeps the fresh single-run / thin-evidence path conservative and bounded through the existing inspect/detail seam.
- The nearest fresh-path probe does not expose a missing runtime branch; the current behavior remains intentionally `Reuse as-is` unless stale freshness or stronger compare evidence justifies tighter guidance.
- No production change is warranted for this slice right now; treat further work here as guard coverage only if a sharper fresh-path requirement appears.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_missing_evidence_reason_for_single_run -q`
- result: `1 passed`

### Slice 2 — Fresh/stale symmetry guard on detail surfaces

**Status:** implemented

**Intent:** Zablokować drift między shared detail widget i template detail widget dla pełnego bounded cue set w obu wariantach freshness.

**Good shape:**
- likely test-only,
- assert `Decision`, `Recommended next action`, provenance, and run summary together,
- no runtime refactor unless divergence is real.

**Implemented:**
- Tightened `tests/test_workspace_history_controller.py` with one compact fresh single-run parity guard across both detail surfaces.
- The new guard now asserts the same bounded cue set on shared and template detail surfaces for the recent/thin-evidence path: `Decision == Reuse as-is`, `Decision based on limited run evidence`, `Validation freshness: recent`, and `Recommended next action == Evidence: only one run available`.
- The probe stayed test-only because the existing runtime already mirrored the fresh-path cues correctly.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_keeps_fresh_single_run_cues_aligned_across_detail_surfaces -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q`
- result: `67 passed`
- `.venv/bin/ruff check tests/test_workspace_history_controller.py`
- result: `All checks passed`

### Slice 3 — Reuse-vs-refine wording refinement after recent evidence

**Status:** covered by existing behavior

**Intent:** Jeżeli prompt ma świeże, ale nadal ograniczone evidence, operator powinien dostać bardziej czytelny bounded next step niż obecne neutralne fallback wording.

**Good shape:**
- keep `Decision` conservative,
- refine only next-action wording,
- no extra workflow branch.

**Probe result:**
- The current inspect/detail seam already keeps the recent thin-evidence path intentionally conservative: fresh single-run prompts stay on `Decision == Reuse as-is` with `Recommended next action == Evidence: only one run available`.
- The paired fresh/stale probes do not show an ambiguous reuse-vs-refine branch that needs another wording layer; the stale path already tightens to `Validate before reuse`, while the recent path avoids overconfident guidance.
- No runtime wording change is warranted for this slice right now; treat further work here as a future product-choice question, not an implementation gap.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_keeps_fresh_single_run_cues_aligned_across_detail_surfaces tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_missing_evidence_reason_for_single_run -q`
- result: `2 passed`

### Slice 4 — Workspace handoff coherence for fresh evidence

**Status:** covered by existing behavior

**Intent:** Skoro stale handoff hint już istnieje, sprawdzić czy ścieżka dla świeżego evidence nie potrzebuje równie czytelnego, ale lekkiego hintu przy `Open in Workspace`.

**Good shape:**
- reuse `PromptActionsController.open_prompt_in_workspace()` seam,
- keep existing toast semantics,
- add only one bounded fresh-path hint if RED proves the gap is real.

**Probe result:**
- The current workspace handoff already has a bounded fresh-path hint: `Prompt ready in workspace. Run current prompt to validate before refining.`
- The paired handoff probes show the intended asymmetry is already present: fresh evidence keeps the generic validate-before-refine guidance, while stale evidence upgrades only the status hint to the stronger stale-validation wording.
- No additional fresh-path runtime wording is warranted right now; changing it further would be a product-choice change, not a missing implementation seam.

**Verified:**
- `.venv/bin/pytest tests/test_prompt_actions_controller.py::test_open_prompt_in_workspace_seeds_text_without_running tests/test_prompt_actions_controller.py::test_open_prompt_in_workspace_surfaces_stale_validation_handoff_hint -q`
- result: `2 passed`

### Slice 5 — Headless parity decision for any newly shared cue

**Status:** CLI unchanged by design

**Intent:** Każdy nowy bounded semantic z tego cyklu musi przejść przez regułę shared-vs-local.

**Decision:**
- The only newly probed cue in this cycle's workspace branch remains the handoff status emitted by `PromptActionsController.open_prompt_in_workspace()`.
- That wording is action-local and does not flow through shared analytics fields such as `decision_summary`, `next_action_summary`, or `freshness_summary`.
- The existing `history-analytics` CLI path already renders only shared execution-summary semantics, so adding the workspace handoff hint there would create a shadow headless model instead of exposing shared product truth.

**Deliberate no-op:**
- Keep `core/history_tracker.py` unchanged.
- Keep `cli/commands.py` unchanged.
- Keep `tests/test_main_entry.py` unchanged.
- Revisit only if a future slice promotes this semantic onto an existing shared analytics field.

**Verified:**
- inspected `tests/test_main_entry.py` `history-analytics` expectations for shared `decision`, `next`, and `freshness` lines
- confirmed the current roadmap cycle introduced no new shared analytics field beyond already-shipped CLI parity
- result: `CLI unchanged by design`

### Slice 6 — Operator-path guard pack

**Status:** implemented

**Intent:** Po 1–2 runtime slices domknąć najbliższe ryzyka lekkimi guardami zamiast rozszerzać funkcje.

**Good shape:**
- selection clear/reset,
- duplicate-suppression regression,
- shared/template parity,
- fresh-vs-stale wording symmetry.

**Implemented:**
- Added one focused widget-level regression guard for the duplicate-suppression path on the shared detail surface.
- The new probe locks the sequence where `Recommended next action` is hidden because it duplicates `Decision`, then briefly re-exposed when the decision clears, and finally suppressed again when the same decision returns.
- The runtime already behaved correctly, so this slice landed as test-only coverage with no production-code changes.

**Verified:**
- `.venv/bin/pytest tests/test_prompt_detail_widget.py::test_prompt_detail_widget_keeps_hidden_duplicate_next_action_suppressed_when_same_decision_returns -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_detail_widget.py tests/test_workspace_history_controller.py -q`
- result: `56 passed`
- `.venv/bin/ruff check tests/test_prompt_detail_widget.py tests/test_workspace_history_controller.py`
- result: `All checks passed!`

---

## Recommended first slice

### Pick first: Slice 1 — Fresh-validation next-step cue

Why this first:
- directly continues the existing stale-validation work without inventing a new product track,
- uses seams already proven in `WorkspaceHistoryController`,
- has an obvious RED test path,
- can stay entirely bounded to wording / operator guidance,
- should reveal quickly whether runtime change is needed or whether this is another test-only guard.

Why not start with workspace first:
- workspace handoff is a secondary seam,
- inspect/detail remains the canonical decision-support surface,
- starting in workspace risks overfitting action-local wording before the shared operator path is settled.

---

## Implementation brief for the first slice

### Task 1: Confirm current fresh-path behavior

**Objective:** Inspect the exact current output for recent single-run / thin-evidence prompts before writing assertions.

**Files:**
- Read: `gui/workspace_history_controller.py`
- Read: `tests/test_workspace_history_controller.py`
- Maybe read: `tests/test_prompt_detail_widget.py`

**Verification:**
- identify the existing recent-freshness test seam,
- confirm whether a new RED is likely or whether the behavior is already covered.

### Task 2: Write failing test for fresh-validation next action

**Objective:** Prove the current inspect/detail seam does not yet express the desired fresh-path guidance.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`
- Maybe modify: `tests/test_prompt_detail_widget.py`

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q`
- Expected before implementation: one failing assertion on next-action wording or symmetry expectation

### Task 3: Implement minimal wording refinement only if RED is real

**Objective:** Tighten the existing next-action seam for the recent/fresh evidence path without changing stronger decision branches.

**Files:**
- Modify: `gui/workspace_history_controller.py`

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q`
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`

### Task 4: If RED passes immediately, convert the slice into a guard-only closure

**Objective:** Avoid fake runtime churn when the current seam already behaves correctly.

**Files:**
- Modify: this roadmap
- Modify: `docs/CHANGELOG.md` only if a meaningful test-only ledger note is warranted

**Verification:**
- record `covered by existing behavior` with the exact pytest command/result
- note explicitly whether the probe found a real fresh-path gap or confirmed the current conservative behavior
- for parity slices, prefer marking the slice `implemented` when new regression coverage materially strengthens the execution ledger even if production code stays unchanged

### Task 5: Decide on headless parity

**Objective:** Apply the shared-vs-local rule to any wording that survives the slice.

**Files:**
- Maybe modify: `tests/test_main_entry.py`
- Maybe modify: `cli/commands.py`
- Maybe modify: `core/history_tracker.py`

**Verification:**
- `.venv/bin/pytest tests/test_main_entry.py -q` if touched
- otherwise document `CLI unchanged by design`

### Task 6: Sync roadmap ledger after green

**Objective:** Keep this roadmap as the canonical execution ledger for the new cycle.

**Files:**
- Modify: this file
- Modify: `docs/CHANGELOG.md`
- Update `docs/product-ssot.md` only if product posture actually changes

**Required notes:**
- what changed,
- whether the slice landed as runtime or test-only,
- exact verification commands/results,
- whether CLI changed or intentionally did not.

---

## Documentation rule for this roadmap

Po każdej implementacji w tym cyklu:
1. najpierw zaktualizować ten roadmap,
2. dopisać krótki implemented / verified note,
3. zaktualizować `docs/CHANGELOG.md` jeśli zmieniło się user-visible behavior albo istotny execution ledger,
4. aktualizować `docs/product-ssot.md` tylko jeśli zmieni się produktowy priorytet albo definicja warstwy.

SSOT ma pozostać stabilny; ten plik ma być żywym ledgerem wykonania.

---

## Definition of done for this cycle

Ten cykl będzie można uznać za dobrze domknięty, gdy PromptManager:
- lepiej komunikuje kolejny bounded krok po obejrzeniu run evidence,
- utrzymuje spójność między inspect/detail i workspace handoff,
- zachowuje asset-first posture,
- nie tworzy nowego heavy workflow,
- oraz ma jasną regułę, które semantics są shared i dostają parity, a które pozostają local by design.
