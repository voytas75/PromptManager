# PromptManager Run-Evidence Legibility Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć następny execution cycle po domknięciu asset-to-run-to-refine coherence v3 i poprawić czytelność samego run evidence, tak żeby operator szybciej rozumiał co ostatni run naprawdę mówi o prompt asset bez dokładania dashboardów, nowych workflowów ani shadow modelu.

**Architecture:** Ten roadmap startuje po zamknięciu roadmap z 2026-04-25 i 2026-04-26. Nie wraca do settings ani governance foundation. Nowy cykl skupia się na tym, jak istniejące inspect/detail i history-analytics pokazują ostatni run, jego comparability, provenance i practical reading order. Priorytetem jest legibility existing evidence, nie nowa warstwa oceniania.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, existing PromptManager execution history, inspect/detail widgets, workspace history controller, shared analytics / CLI rendering, docs under `docs/`.

---

## Why this roadmap exists

Live repo review confirms:
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` is fully implemented in scope,
- `docs/plans/2026-04-26-next-cycle-roadmap.md` is fully implemented in scope,
- `docs/plans/2026-04-26-asset-to-run-to-refine-coherence-v3-roadmap.md` is now exhausted in planned scope,
- `docs/product-ssot.md` still prioritizes prompt assets first, trustworthy runs second, automation third,
- current Unreleased changelog already records the delivered bounded next-action / freshness / workspace-handoff / parity work.

So the next useful move is not another handoff micro-slice, but a fresh bounded cycle that improves how operators read existing run evidence itself.

Short version:

**assets first -> trustworthy runs -> asset-to-run-to-refine coherence -> run-evidence legibility**

---

## Confirmed baseline before this cycle

Assume already delivered:
- deterministic settings, routing, and trust diagnostics,
- prompt-linked structured runs and bounded comparison evidence,
- inspect/detail decision, next-action, provenance, and freshness cues,
- stale-validation-aware workspace handoff messaging,
- shared/template detail parity for the main bounded cues,
- CLI/headless parity only for shared execution-summary semantics,
- operator-path guards for clear/reset and duplicate-suppression regressions.

This cycle should not rebuild those layers.
It should make existing run evidence easier to read and judge at a glance.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager już umie dać bounded decision cue, to jak sprawić, żeby sam evidence summary był bardziej legible i mniej "operator musi sobie dopowiedzieć"?

Nowy cykl wzmacnia trzy rzeczy:

1. **run-summary legibility** — operator szybciej rozumie, co było uruchomione, czy to było comparable i czy w ogóle warto wyciągać wniosek;
2. **evidence-reading posture** — inspect/detail i headless surfaces mają pomagać czytać evidence we właściwej kolejności, zamiast tylko dopisywać kolejne cues;
3. **shared semantics discipline** — jeśli legibility improvement dotyczy shared execution summary, parity jest obowiązkowe; jeśli dotyczy tylko widget-local formatting, trzeba to jasno ograniczyć.

---

## Constraints

Do not add:
- a new dashboard,
- a new benchmark panel,
- a scoring engine,
- a review queue,
- new persistence only for summary wording,
- background recomputation,
- a CLI-only shadow evidence model.

Reuse first:
- `gui/workspace_history_controller.py`
- `gui/widgets/prompt_detail_widget.py`
- `core/history_tracker.py`
- `cli/commands.py`
- `tests/test_workspace_history_controller.py`
- `tests/test_prompt_detail_widget.py`
- `tests/test_main_entry.py`

---

## Roadmap stages for this cycle

### Stage A — Last-run summary readability

Cel: poprawić czytelność tego, co już jest w run summary bez zmiany modelu danych.

Obszary:
- ordering of summary fragments,
- compact wording for comparable vs non-comparable evidence,
- consistency between recent/stale and successful/weak evidence reading.

Constraint:
- prefer wording/order changes over new fields,
- no extra labels if current summary seam can carry the intent.

### Stage B — Comparable-evidence legibility

Cel: zmniejszyć niejasność, kiedy run evidence wygląda "pełnie", ale tak naprawdę nie nadaje się jeszcze do sensownego compare reading.

Obszary:
- compact comparability cue,
- bounded non-comparable fallback wording,
- avoid overclaiming compare posture from thin evidence.

Constraint:
- do not build a new compare workflow,
- reuse existing analytics/history facts only.

### Stage C — Shared evidence reading order

Cel: jeśli to samo evidence jest shared, GUI i headless powinny prowadzić operatora podobną logiką czytania.

Obszary:
- parity for shared run-summary semantics,
- rendering order between summary / decision / next-step only where justified,
- avoid one surface implying stronger certainty than another.

Constraint:
- GUI remains primary interactive surface,
- CLI only mirrors shared truth.

### Stage D — Regression guard pack for evidence wording

Cel: po 1–2 małych zmianach domknąć drift risk lekkimi testami zamiast dalszej rozbudowy produktu.

Obszary:
- shared/template parity,
- headless parity,
- summary ordering guards,
- negative guards for duplicate or misleading wording.

Constraint:
- prefer test-only slices whenever runtime already behaves correctly.

---

## Recommended execution order

1. run-summary readability on shared inspect/detail surfaces
2. comparability / non-comparability cue discipline
3. shared GUI/CLI parity for any shared evidence wording
4. guard-pack closure

---

## Candidate bounded slices

### Slice 1 — Last-run summary ordering probe

**Status:** implemented

**Intent:** Sprawdzić, czy obecny `run_summary` pokazuje najważniejszy evidence fragment w najlepszej kolejności dla operatora.

**Implemented:**
- added one focused RED/green probe in `tests/test_workspace_history_controller.py` for the shared/template detail run-summary order,
- tightened `WorkspaceHistoryController._build_run_summary()` so `via <model>` now stays attached to the leading `Last run: <status>` fact instead of rendering as a detached middle fragment,
- kept comparability evidence after core last-run facts and freshness, with no new fields or persistence.

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_places_candidate_vs_baseline_after_last_run_facts -q` → failed before runtime fix, passed after fix
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q` → `23 passed`
- `.venv/bin/pytest tests/test_prompt_detail_widget.py tests/test_workspace_history_controller.py -q` → `57 passed`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py` → `All checks passed!`

**CLI parity:** unchanged by design — this slice adjusted widget-local run-summary wording/order only and did not change shared headless execution-summary semantics.

**Examples:**
- `Last run: success via gpt-4o-mini · v3 · 2 messages · 140 ms · Validation freshness: recent`
- comparison evidence remains appended after those last-run facts when present.

### Slice 2 — Comparable-evidence cue in run summary

**Status:** implemented

**Intent:** Jeżeli istniejące history facts pozwalają rozpoznać, że latest run nie jest jeszcze sensownie comparable, operator powinien widzieć to bliżej samego run evidence, nie tylko w późniejszym decision cue.

**Good shape:**
- reuse existing execution-history facts,
- compact wording,
- absent when evidence is already clearly comparable,
- no scoring language.

**Implemented:**
- extended `WorkspaceHistoryController._build_run_summary()` with one bounded comparison-readiness seam before the existing `Candidate vs baseline` summary,
- added `Comparison readiness: no baseline yet` when the two newest runs still share the same prompt version, so the inspect path exposes non-comparable evidence directly in the run summary,
- added `Comparison readiness: limited` when two-version evidence exists but rating or duration metadata is still missing, while keeping the cue absent once evidence is fully comparable,
- tightened controller and widget coverage so both shared and template detail surfaces render the new readiness wording without changing persistence or scoring semantics.

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_missing_evidence_reason_for_non_comparable_baseline -q` → failed before runtime change, passed after fix
- `.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_missing_evidence_reason_for_missing_rating tests/test_prompt_detail_widget.py::test_prompt_detail_widget_renders_last_run_summary_label -q` → `2 passed`
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q` → `56 passed`
- `.venv/bin/pytest tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q` → `12 passed`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py` → `All checks passed!`

### Slice 3 — Headless parity for shared run-evidence wording

**Status:** CLI unchanged by design

**Intent:** Jeśli Slice 1 albo 2 zmienia shared execution-summary meaning, `history-analytics` musi to odzwierciedlić.

**Decision:**
- Slice 1 changed only widget-local run-summary ordering/legibility and already documented that CLI parity remained unchanged by design.
- Slice 2 added `Comparison readiness: ...` only on the inspect/detail `run_summary` seam in `WorkspaceHistoryController`; it did not promote that wording into shared analytics fields such as `decision_summary`, `next_action_summary`, or `freshness_summary`.
- The existing `history-analytics` CLI path still correctly renders only shared execution-summary semantics, so adding `Comparison readiness` there would create a shadow headless model instead of exposing shared product truth.

**Deliberate no-op:**
- keep `core/history_tracker.py` unchanged,
- keep `cli/commands.py` unchanged,
- keep `tests/test_main_entry.py` unchanged.

**Verified:**
- inspected `core/history_tracker.py` shared analytics payload and confirmed it still exposes only `decision_summary`, `next_action_summary`, and `freshness_summary`
- inspected `cli/commands.py` `history-analytics` rendering and confirmed it still prints only those shared fields
- inspected `tests/test_main_entry.py::test_history_analytics_command_renders_summary` and confirmed the CLI expectation remains aligned with shared semantics only
- result: `CLI unchanged by design`

### Slice 4 — Evidence-reading guard pack

**Status:** implemented

**Intent:** Domknąć ten cykl lekkimi guardami na wording drift i parity drift.

**Good shape:**
- shared/template parity,
- CLI parity or deliberate no-op,
- ordering regression guard,
- no new production logic unless RED exposes a real bug.

**Implemented:**
- The cycle already landed the expected guard shape across the delivered slices instead of as one separate runtime change.
- Slice 1 locked the operator-facing `Last run: ...` ordering on the existing inspect/detail seam.
- Slice 2 added explicit shared/template parity coverage for `Comparison readiness: no baseline yet` and `Comparison readiness: limited` on both detail surfaces.
- Slice 3 closed the headless branch as a deliberate no-op, keeping CLI parity restricted to shared analytics semantics only.
- No further production logic was required because the remaining drift risk was already covered by the shipped controller/widget tests and the documented CLI no-op rule.

**Verified:**
- `tests/test_workspace_history_controller.py` asserts shared/template parity for both `Comparison readiness: no baseline yet` and `Comparison readiness: limited`
- `tests/test_prompt_detail_widget.py::test_prompt_detail_widget_renders_last_run_summary_label` locks the compact last-run wording, including freshness and limited-readiness rendering
- `docs/plans/2026-04-26-asset-to-run-to-refine-coherence-v3-roadmap.md` carries forward the same shared-vs-local CLI rule for later slices
- result: `implemented via earlier slices and guard coverage`

---

## Recommended first slice

### Pick first: Slice 1 — Last-run summary ordering probe

Why this first:
- directly strengthens evidence legibility without inventing a new semantic layer,
- stays close to the asset-to-run-to-refine loop instead of broadening the product,
- has an obvious TDD/probe path,
- should quickly tell us whether a real runtime improvement exists or whether current ordering is already good enough.

Why not start with comparability first:
- comparability wording risks adding another semantic before we confirm the basic run summary is readable,
- ordering/readability is cheaper and lower-risk as the next probe.

---

## Implementation brief for the first slice

### Task 1: Inspect current run-summary construction

**Objective:** Confirm the current fragment order and identify the exact test seam that owns it.

**Files:**
- Read: `gui/workspace_history_controller.py`
- Read: `tests/test_workspace_history_controller.py`
- Maybe read: `core/history_tracker.py`

**Verification:**
- identify where `Last run: ...` strings are built,
- confirm whether current order is already deliberate and tested.

### Task 2: Add a focused RED test for summary order or readability

**Objective:** Prove the current run-summary output is suboptimal before changing runtime behavior.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`
- Maybe modify: `tests/test_main_entry.py` only if a shared field is involved immediately

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q`
- Expected before implementation: one failing assertion on run-summary wording/order if there is a real gap.

### Task 3: Implement the minimal change only if RED is real

**Objective:** Adjust ordering or wording on the existing summary seam without changing stored analytics structure unless necessary.

**Files:**
- Modify: `gui/workspace_history_controller.py`
- Maybe modify: `core/history_tracker.py` only if the change is truly shared

**Verification:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py -q`
- `.venv/bin/pytest tests/test_prompt_detail_widget.py tests/test_workspace_history_controller.py -q`

### Task 4: Decide on headless parity

**Objective:** Apply the shared-vs-local rule immediately after green.

**Files:**
- Maybe modify: `tests/test_main_entry.py`
- Maybe modify: `cli/commands.py`
- Maybe modify: `core/history_tracker.py`

**Verification:**
- `.venv/bin/pytest tests/test_main_entry.py -q` if touched
- otherwise document `CLI unchanged by design`

### Task 5: Sync the execution ledger

**Objective:** Keep this roadmap as the canonical ledger for the new cycle.

**Files:**
- Modify: this file
- Modify: `docs/CHANGELOG.md` if behavior or meaningful guard coverage changed
- Update `docs/product-ssot.md` only if product posture changes

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
3. zaktualizować `docs/CHANGELOG.md` jeśli zmienił się user-visible behavior albo execution ledger,
4. aktualizować `docs/product-ssot.md` tylko jeśli zmieni się produktowy priorytet albo warstwa produktu.

SSOT ma pozostać stabilny; ten plik ma być żywym ledgerem wykonania.

---

## Definition of done for this cycle

Ten cykl będzie można uznać za dobrze domknięty, gdy PromptManager:
- lepiej komunikuje, jak czytać ostatni run evidence,
- nie myli bounded decision cues z samym evidence summary,
- utrzymuje shared semantics tam, gdzie to naprawdę shared,
- pozostaje asset-first i nie skręca w analytics-first UX,
- oraz ma domknięte guardy przeciw wording/parity drift.
