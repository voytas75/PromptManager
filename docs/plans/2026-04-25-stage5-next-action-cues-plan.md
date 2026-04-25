# PromptManager Stage 5 — Next-Action Cues Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Wzmocnić core prompt loop przez dodanie jednego bounded, user-visible `Recommended next action` cue w inspect/detail, opartego wyłącznie na już istniejących sygnałach lineage i run evidence.

**Architecture:** Ten slice powinien rozszerzyć istniejący inspect/detail seam zamiast tworzyć nowy model produktu. Źródłem prawdy pozostaje `WorkspaceHistoryController`, który już buduje `decision_summary` i `run_summary`; nowy cue ma tylko syntetyzować istniejące decyzje i evidence do bardziej operacyjnego next step. Brak nowej persystencji, brak nowych ekranów, brak równoległej warstwy API.

**Tech Stack:** Python 3.13, PySide6, pytest, existing inspect/detail widgets and workspace history controller, docs under `docs/`.

---

## Docs / SSOT review outcome

### Potwierdzone z SSOT
- `docs/product-ssot.md` utrzymuje zasadę: **assets first, operations second, automation third**.
- Core loop nadal kończy się na: **Inspect → Reuse → Refine → Run**.
- W `In scope — core` szczególnie istotne są teraz:
  - `Inspection clarity`
  - `Reuse and refinement`
  - `compact decision-support cues`
- `docs/product-roadmap-ssot.md` ustawia po Stage 4 wejście w **Stage 5 — Selective expansion**, ale nadal z filtrem: wzmacniać główną pętlę produktu, nie iść w dashboard-first ani broad automation.
- `docs/plans/2026-04-25-roadmap-implementation-plan.md` ma zamknięte Stage 1–4; nie ma jeszcze rozpisanego pierwszego bounded slice’a dla Stage 5.

### Wniosek produktowy
Najbardziej spójny następny krok to **mały inspect/reuse slice**, nie nowa automatyzacja ani analytics surface. Najniższe ryzyko i najwyższa zgodność z SSOT daje dołożenie `Recommended next action` do już istniejącego inspect/detail flow.

### Existing seam review
- `gui/workspace_history_controller.py` już buduje:
  - `decision_summary` (`Reuse as-is`, `Fork before editing`, `Refine before reuse`, `Safe to compare`)
  - `run_summary` (`Last run`, `Candidate vs baseline`)
- `tests/test_workspace_history_controller.py` już pokrywa te bounded cues.
- `tests/test_prompt_detail_widget.py` już ma osobne pokrycie dla label rendering `Decision:` i `Last run:`.

### Design rule for this slice
Nowy cue nie może dublować starego `Decision:` 1:1 bez żadnej wartości. Musi być bardziej operacyjny: mówić użytkownikowi **co zrobić teraz**, a nie tylko **jak sklasyfikować stan**. Ma jednak bazować wyłącznie na obecnych sygnałach.

---

## Proposed product behavior

Dodaj nowy kompaktowy cue w inspect/detail:

- `Recommended next action: Compare before promoting`
- `Recommended next action: Refine candidate`
- `Recommended next action: Fork before editing`
- `Recommended next action: Reuse as-is`

### Mapping v1
- jeśli `decision_summary == "Safe to compare"` → `Compare before promoting`
- jeśli `decision_summary == "Refine before reuse"` → `Refine candidate`
- jeśli `decision_summary == "Fork before editing"` → `Fork before editing`
- w pozostałych przypadkach → `Reuse as-is`

### Why this mapping
- zachowuje bounded charakter
- nie wymaga nowej logiki domenowej ani scoringu
- wykorzystuje już istniejące inspect/recommendation seams
- wzmacnia user-visible actionability w centrum produktu

### Non-goals
- brak nowej historii akcji
- brak CTA button routing
- brak zmian w storage / models
- brak zmian w search list / catalog view
- brak nowych heurystyk jakości promptu

---

## Task 1: Add failing widget test for next-action cue rendering

**Objective:** Udowodnić, że shared detail widget potrafi renderować nowy cue niezależnie od kontrolera.

**Files:**
- Modify: `tests/test_prompt_detail_widget.py`
- Later modify: `gui/widgets.py` or the exact shared detail widget module where `PromptDetailWidget` lives

**Step 1: Write failing test**

Add a new widget-level test near the existing `Decision:` / `Last run:` tests:

```python
def test_prompt_detail_widget_renders_next_action_summary_label(
    qt_app: QApplication,
) -> None:
    """Detail view should render one compact next-action cue when the controller provides it."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_next_action_summary("Compare before promoting")
    qt_app.processEvents()

    assert widget._next_action_label.isVisible()  # noqa: SLF001
    next_action_text = widget._next_action_label.text()  # noqa: SLF001
    assert "Recommended next action:" in next_action_text
    assert "Compare before promoting" in next_action_text
```

**Step 2: Run test to verify failure**

Run:
```bash
.venv/bin/pytest tests/test_prompt_detail_widget.py::test_prompt_detail_widget_renders_next_action_summary_label -q
```

Expected: FAIL because the widget does not yet expose `update_next_action_summary` / `_next_action_label`.

**Step 3: Implement minimal widget support**

In the shared detail widget:
- add one hidden-by-default label for next action
- keep wording compact and high-contrast
- add `update_next_action_summary(text: str | None)` with the same show/hide pattern as decision/run labels

**Step 4: Run test to verify pass**

Run the same command again.

**Step 5: Commit**

```bash
git add tests/test_prompt_detail_widget.py gui/widgets.py
git commit -m "feat: add next-action cue to prompt detail widget"
```

---

## Task 2: Add failing controller test for inspect/detail next-action mapping

**Objective:** Zakotwiczyć nowy cue w istniejącym decision/evidence seam bez nowego modelu.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`
- Modify later: `gui/workspace_history_controller.py`

**Step 1: Extend the detail widget stub**

In `_PromptDetailWidgetStub`, add storage for:

```python
self.next_action_summary: str | None = None
```

and a setter:

```python
def update_next_action_summary(self, text: str | None) -> None:
    self.next_action_summary = text
```

Also clear it in `clear()`.

**Step 2: Write failing controller test for comparison path**

Add a test near the existing `Safe to compare` coverage:

```python
def test_workspace_history_controller_surfaces_compare_before_promoting_next_action_for_compatible_runs() -> None:
    ...
    controller.handle_selection_changed()
    assert detail_widget.next_action_summary == "Compare before promoting"
    assert template_detail_widget.next_action_summary == "Compare before promoting"
```

Reuse the same compatible two-run fixture shape already used by the `Safe to compare` test.

**Step 3: Run targeted test to verify failure**

Run:
```bash
.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_compare_before_promoting_next_action_for_compatible_runs -q
```

Expected: FAIL because controller does not yet calculate or push the next-action cue.

**Step 4: Implement minimal controller support**

In `gui/workspace_history_controller.py`:
- call a new `_update_prompt_next_action_summary(prompt)` from `handle_selection_changed()`
- add `_build_next_action_summary(prompt)`
- in v1, map from existing `decision_summary` values only

Suggested bounded implementation shape:

```python
def _build_next_action_summary(self, prompt: Prompt) -> str:
    decision = self._build_decision_summary(prompt)
    if decision == "Safe to compare":
        return "Compare before promoting"
    if decision == "Refine before reuse":
        return "Refine candidate"
    if decision == "Fork before editing":
        return "Fork before editing"
    return "Reuse as-is"
```

**Step 5: Run targeted test to verify pass**

Run the same command again.

**Step 6: Commit**

```bash
git add tests/test_workspace_history_controller.py gui/workspace_history_controller.py
git commit -m "feat: map inspect evidence to next-action cues"
```

---

## Task 3: Add fallback-path coverage for lineage-only prompts

**Objective:** Upewnić się, że next action działa nie tylko dla compare path, ale też dla obecnych lineage decisions.

**Files:**
- Modify: `tests/test_workspace_history_controller.py`

**Step 1: Add two small assertions/tests**

Either extend existing tests or add dedicated ones asserting:
- `Refine before reuse` → `Refine candidate`
- `Fork before editing` → `Fork before editing`
- `Reuse as-is` → `Reuse as-is`

Prefer separate focused assertions if existing tests stay readable.

**Step 2: Run focused suite**

```bash
.venv/bin/pytest tests/test_workspace_history_controller.py -q
```

Expected: green after minimal controller implementation.

**Step 3: Commit**

```bash
git add tests/test_workspace_history_controller.py
git commit -m "test: cover fallback next-action mappings"
```

---

## Task 4: Broader inspect/detail smoke

**Objective:** Sprawdzić, że nowy cue nie psuje istniejących bounded inspect surfaces.

**Files:**
- No new files required unless smoke exposes a bug

**Step 1: Run nearby widget/controller smoke**

```bash
.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q
```

Expected: PASS

**Step 2: Run adjacent inspect-path smoke**

```bash
.venv/bin/pytest tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q
```

Expected: PASS

**Step 3: Optional syntax sanity check**

```bash
python -m py_compile gui/workspace_history_controller.py
```

Expected: no output

---

## Task 5: Update roadmap plan and product docs only if scope changes

**Objective:** Zamknąć slice zgodnie z canonical workflow rule.

**Files:**
- Modify: `docs/plans/2026-04-25-roadmap-implementation-plan.md`
- Create or maintain: this plan file `docs/plans/2026-04-25-stage5-next-action-cues-plan.md`
- Update only if wording truly changes: `README.md`, `docs/product-ssot.md`, `docs/product-roadmap-ssot.md`

**Step 1: Update roadmap implementation plan**

Add a new Stage 5 entry or update log entries with:
- `Status: implemented`
- `Implemented:` bullets
- `Verified:` bullets with exact commands/results

**Step 2: Re-check SSOT need honestly**

Update SSOT only if the product wording itself changed.

Current expectation:
- `docs/product-ssot.md`: probably **no change needed**
- `docs/product-roadmap-ssot.md`: probably **no change needed**
- reason: this slice fits the already-stated core of `inspection clarity` and `compact decision-support cues`

**Step 3: Commit docs**

```bash
git add docs/plans/2026-04-25-roadmap-implementation-plan.md docs/plans/2026-04-25-stage5-next-action-cues-plan.md
git commit -m "docs: record stage 5 next-action cue plan and progress"
```

---

## Recommended execution order

1. widget rendering seam
2. controller compare-path mapping
3. fallback mapping coverage
4. smoke tests
5. roadmap/doc updates

---

## Verification checklist

### Targeted
```bash
.venv/bin/pytest tests/test_prompt_detail_widget.py::test_prompt_detail_widget_renders_next_action_summary_label -q
.venv/bin/pytest tests/test_workspace_history_controller.py::test_workspace_history_controller_surfaces_compare_before_promoting_next_action_for_compatible_runs -q
```

### Slice-level
```bash
.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q
```

### Nearby smoke
```bash
.venv/bin/pytest tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q
```

### Syntax sanity
```bash
python -m py_compile gui/workspace_history_controller.py
```

---

## Exit criteria

This slice is done when:
- inspect/detail shows one compact `Recommended next action:` cue
- cue is derived only from existing bounded lineage/run evidence
- no new persistence or product model is introduced
- targeted tests and nearby smoke are green
- roadmap implementation plan is updated

---

## Expected user-visible outcome

Inspect/detail stops at just telling the user what state a prompt is in and starts telling them what to do next, while staying compact, deterministic, and grounded in existing evidence.
