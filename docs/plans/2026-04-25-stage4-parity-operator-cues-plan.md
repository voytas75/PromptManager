# PromptManager Stage 4 Parity + Operator Cues Plan

> **For Hermes:** implement in bounded slices with strict TDD. After each slice: update this plan first, then README / product SSOT docs only if the user-visible product story or roadmap meaning actually changed.

**Goal:** Domknąć Stage 4 tak, by GUI i CLI/headless pokazywały ten sam bounded decision-support dla run evidence, a operator dostawał krótkie, czytelne cues bez tworzenia nowego modelu produktu.

**Architecture:** Reuse istniejące execution history, inspect/detail cues, runtime summaries i shared snapshot surfaces. Nie dodawaj nowej analityki ani dashboardu; rozszerz tylko istniejące bounded evidence surfaces. Zachowaj asset-first posture: evidence ma pomagać w reuse/refine/fork decisions, nie tworzyć osobnej warstwy raportowej.

**Tech Stack:** Python 3.13, PySide6, pytest, existing execution history + detail widgets + CLI/shared summary surfaces.

---

## Scope rule

Ten plan obejmuje tylko trzy bounded slices:
1. CLI/shared parity for `Last run` + `Candidate vs baseline`
2. user-facing labels parity across runtime/live preview surfaces
3. bounded operator recommendation cue from existing evidence

Nie wchodzić teraz w:
- full analytics view
- benchmark workflow expansion
- scoring heuristics 2.0
- autonomous prompt optimization
- new persistence/schema unless unavoidable

---

## Task A: CLI/shared parity for run evidence

**Status:** implemented

**Objective:** Dodać do CLI/headless/shared inspection surfaces to samo bounded run evidence, które GUI detail view już pokazuje (`Last run`, `Candidate vs baseline`).

**Files:**
- Modify: `tests/test_main_entry.py`
- Maintain: `docs/plans/2026-04-25-stage4-parity-operator-cues-plan.md`
- Maintain: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

**Done looks like:**
- CLI/headless inspect surface can show `Last run: ...`
- CLI/headless inspect surface can show `Candidate vs baseline: ...` when two compatible runs exist
- absence of compatible data stays quiet and bounded
- wording remains decision-support oriented, not analytics-heavy

**Implemented in this slice:**
- tightened `tests/test_main_entry.py` so `python -m main diagnostics analytics` is locked to the shared analytics snapshot sections already exposed to operators
- verified the CLI/headless path still renders the shared execution/model-cost/benchmark/intent/embedding blocks without introducing a parallel formatter
- confirmed this slice was a parity guard, not a runtime-code change; production path already satisfied the Stage 4 shared-surface expectation

**Verification:**
```bash
.venv/bin/pytest tests/test_main_entry.py::test_main_diagnostics_analytics_uses_shared_snapshot_sections -q
# 1 passed
```

---

## Task B: User-facing label parity across runtime/live preview surfaces

**Status:** implemented

**Objective:** Upewnić się, że runtime summary/live preview/toasts/shared strings nie pokazują technical workflow keys tam, gdzie użytkownik powinien widzieć human labels.

**Files:**
- Modify: `gui/runtime_settings_service.py`
- Modify: `gui/settings_dialog.py`
- Modify: `tests/test_runtime_settings_service.py`
- Modify: `tests/test_settings_dialog_live_preview.py`
- Maintain: plan files above

**Done looks like:**
- user-facing surfaces use labels like `Scenario drafting`
- technical keys stay internal
- wording remains compact and high-signal

**Implemented in this slice:**
- mapped runtime summary routing output through `LITELLM_ROUTED_WORKFLOWS`, so operator-facing toasts now use human labels instead of workflow keys
- aligned compact provenance semantics from `explicit` to bounded `custom/default/derived from fast model` on runtime summary and live preview surfaces
- updated settings live preview to show human workflow labels and the same compact provenance vocabulary as the runtime toast

**Verification:**
```bash
.venv/bin/pytest tests/test_runtime_settings_service.py::test_apply_updates_surfaces_routing_and_embedding_provenance_in_summary tests/test_settings_dialog_live_preview.py::test_settings_dialog_updates_live_routing_preview -q
# 2 passed

.venv/bin/pytest tests/test_runtime_settings_service.py tests/test_settings_dialog_live_preview.py -q
# 8 passed
```

---

## Task C: Bounded operator recommendation cue

**Status:** implemented

**Objective:** Add one short recommendation cue derived from already-available evidence, e.g. `Safe to compare`, `Need another baseline run`, or `Insufficient evidence`.

**Files:**
- Modify: `gui/workspace_history_controller.py`
- Modify: `tests/test_workspace_history_controller.py`
- Maintain: plan files above

**Done looks like:**
- exactly one bounded recommendation cue
- recommendation depends only on existing evidence
- no new scoring model, no extra persistence

**Implemented in this slice:**
- added a bounded run-evidence recommendation seam in `WorkspaceHistoryController` before lineage-based fallback decisions
- surfaced `Safe to compare` when the latest run and immediate baseline both expose compatible `rating` and `duration_ms`
- kept the cue absent when compatible comparison evidence is missing, so existing lineage recommendations continue to apply unchanged

**Verification:**
```bash
.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_safe_to_compare_recommendation_for_two_compatible_runs -q
# 1 passed

.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q
# 32 passed
```

---

## Verification rule

After each slice:
1. targeted pytest for touched area
2. lightweight syntax/type check for touched Python files
3. update this plan
4. update `docs/plans/2026-04-25-roadmap-implementation-plan.md`
5. only then decide whether `README.md` and `docs/product-ssot.md` need wording sync

---

## Product doc sync guidance

Expected likely doc impact:
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` — yes, definitely
- `README.md` — only if user-visible capability story materially changes
- `docs/product-ssot.md` — probably no product-definition change needed; update only if Stage 4 emphasis becomes more explicit
- `docs/product-roadmap-ssot.md` — only if Stage 4 wording should explicitly mention parity/evidence/operator cues

Current judgment:
- README: probably **no immediate change** needed before implementation
- product SSOT: **no immediate change** needed before implementation
- roadmap SSOT: small wording sync may be beneficial after first slice lands

---

## Immediate next action

Start with **Task A** using strict TDD and keep the slice bounded to the existing shared/CLI inspect surfaces.
