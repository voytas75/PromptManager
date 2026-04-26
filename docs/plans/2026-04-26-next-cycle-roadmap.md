# PromptManager Next Cycle Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Domknąć kolejny etap PromptManager po ukończeniu Stage 5, przesuwając produkt z bounded inspect cues do bardziej praktycznej warstwy evaluation / governance oraz spójniejszego asset-to-run-to-refine loop.

**Architecture:** Ten roadmap nie resetuje wcześniejszego SSOT. Zakłada, że asset-first core, trust infrastructure, structured runs oraz Stage 5 inspect/detail refinements są już dowiezione. Nowy cykl ma rozwijać tylko to, co wzmacnia decyzje reuse / refine / replace bez tworzenia nowego dashboard-first albo orchestration-first modelu produktu.

**Tech Stack:** Python 3.13, PySide6, pytest, existing PromptManager execution history, inspect/detail surfaces, CLI/headless seams, docs under `docs/`.

---

## Why a new roadmap

Dotychczasowy plan `docs/plans/2026-04-25-roadmap-implementation-plan.md` jest wykonany do końca swojego zakresu.

Nowy cykl powinien:
- zachować asset-first posture,
- wykorzystać istniejące structured runs i inspect/detail cues,
- wejść w następny poziom decyzji operatorskich,
- ale nadal unikać ciężkich dashboardów, nowego persistence modelu i drugiego shadow workflow.

Krótko:

**assets first -> trustworthy runs -> evaluation/governance cues -> dopiero dalsza automatyzacja**

---

## Current baseline before this cycle

Zakładamy jako już dostarczone:
- deterministic settings / routing / diagnostics,
- prompt-linked structured runs,
- bounded run comparison evidence,
- inspect/detail decision + next-action cues,
- compare guidance split into:
  - `Matched baseline`
  - `Compare improved run`
  - `Compare regressed run`
- workspace handoff validation hint,
- parity across shared/template detail surfaces,
- stale compare fallback removed.

To jest punkt startowy dla nowego cyklu.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager już potrafi pokazać, co było uruchamiane i co wygląda lepiej/gorzej, to jak zrobić z tego bardziej praktyczny, ale nadal bounded system oceny i decyzji?

Nowy cykl ma wzmocnić trzy rzeczy:

1. **evaluation posture** — operator ma szybciej rozumieć, czy obecne evidence jest wystarczające;
2. **governance posture** — decyzje reuse/refine/replace mają mieć czytelniejszy powód i lepszy ślad;
3. **asset-to-run-to-refine coherence** — przejście z prompt asset do uruchomienia i z powrotem ma być bardziej legible bez nowego wielkiego workflow.

---

## Roadmap stages for the new cycle

### Stage A — Evidence quality and evaluation posture

Cel: poprawić jakość odpowiedzi na pytanie „czy mam już wystarczające evidence, żeby ufać temu promptowi?”.

Obszary:
- bounded evidence sufficiency cues,
- compact missing-evidence cues,
- comparison readiness / non-readiness wording,
- operator-safe fallbacks when runs are incomplete, stale, or weakly comparable.

Constraint:
- nie budować nowego scoring engine,
- nie dodawać szerokiego analytics dashboardu,
- nie wprowadzać nowej warstwy persistence tylko dla evaluation.

### Stage B — Governance and review posture

Cel: lepiej zakodować decyzje typu reuse / refine / replace / keep baseline bez zmiany produktu w approval system.

Obszary:
- clearer review-oriented wording,
- bounded provenance of why a decision cue appeared,
- compact review status / confidence hints,
- stronger legibility of decision inputs already available in history and lineage.

Constraint:
- bez workflow approvals,
- bez multi-user review system,
- bez rozbudowanych queues/panels.

### Stage C — Asset-to-run-to-refine handoff

Cel: zmniejszyć tarcie między inspect, run, review wyniku i kolejnym krokiem operatorskim.

Obszary:
- clearer handoff cues from detail view into validation path,
- compact result-to-next-step guidance,
- bounded continuity between latest run evidence and reuse/refine action.

Constraint:
- bez nowego orchestration canvas,
- bez rozbijania GUI na nowy moduł procesowy,
- reuse existing inspect/detail/workspace seams first.

### Stage D — Headless parity for evaluation/governance

Cel: jeśli evaluation/guidance rośnie w GUI, najważniejsze bounded semantics muszą być widoczne też na headless surfaces.

Obszary:
- CLI/shared wording parity,
- stable machine-visible summaries where already justified,
- no shadow model divergence.

Constraint:
- GUI remains the primary operator surface,
- CLI/API expose the same product truth, not a different truth.

---

## Recommended execution order

1. evidence sufficiency / missing-evidence posture
2. bounded governance wording and provenance cues
3. inspect-to-run-to-refine handoff coherence
4. CLI/headless parity for the new bounded semantics

---

## Candidate bounded slices

Poniżej jest lista preferowanych mikro-slice na ten cykl. To nie znaczy, że wszystkie muszą być zrobione od razu; to jest kolejność preferencji.

### Slice 1 — Evidence sufficiency cue in inspect/detail

**Status:** implemented

**Intent:** Pokazać, czy obecna liczba/jakość run evidence wystarcza do sensownej decyzji reuse.

**Good shape:**
- jeden kompaktowy cue,
- oparty tylko na istniejących run/history fields,
- absent when evidence is not meaningful,
- no new persistence.

**Examples:**
- `Evidence: enough to validate reuse`
- `Evidence: too thin for reuse decision`

**Implemented:**
- Reused the existing inspect/detail `Recommended next action` seam instead of adding a new label or panel.
- Added one bounded evidence-sufficiency fallback so prompts with run history but without enough comparable evidence now show `Evidence: too thin for reuse decision` instead of falling through to `Reuse as-is`.
- Kept stronger compare-path decisions unchanged when two runs remain comparable, so this slice only tightens thin-evidence behavior.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- result: `45 passed`

### Slice 2 — Missing-evidence reason cue

**Status:** implemented

**Intent:** Jeżeli evidence jest zbyt słabe, operator powinien wiedzieć dlaczego bez czytania surowych danych.

**Good shape:**
- krótki powód typu:
  - only one run,
  - no comparable baseline,
  - missing rating,
  - missing duration,
- bounded wording only,
- reuse existing decision-support seam.

**Implemented:**
- Replaced the generic thin-evidence fallback with more specific bounded reasons on the existing `Recommended next action` seam.
- Inspect/detail now distinguishes `Evidence: only one run available`, `Evidence: no comparable baseline yet`, `Evidence: missing rating for comparison`, and `Evidence: missing duration for comparison`.
- Kept the surface compact and driven only by existing execution history metadata, without adding new storage, UI sections, or analytics logic.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py`

### Slice 3 — Bounded decision provenance cue

**Status:** implemented

**Intent:** Wzmocnić governance posture przez pokazanie, skąd wzięła się obecna rekomendacja.

**Good shape:**
- compact provenance sentence,
- built from already available lineage/run evidence,
- visible only when it materially helps.

**Examples:**
- `Decision based on latest 2 comparable runs`
- `Decision based on fork lineage only`

**Implemented:**
- Added one compact provenance cue beneath the existing decision summary on shared inspect/detail surfaces.
- Decisions derived from comparable runs now explain themselves with `Decision based on latest 2 comparable runs`.
- Decisions derived from fork state now explain themselves with `Decision based on fork lineage only`, without introducing a review panel or new evidence model.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- `.venv/bin/ruff check gui/workspace_history_controller.py gui/widgets/prompt_detail_widget.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py`

### Slice 4 — Replace-path decision cue

**Status:** implemented

**Intent:** Obecne cues dobrze rozróżniają reuse/refine/fork/compare, ale nie ma jeszcze bounded replace posture, jeśli candidate wyraźnie przegrywa.

**Good shape:**
- very narrow trigger,
- only if existing run evidence is strong enough,
- compact guidance, no new workflow branch.

**Examples:**
- `Keep baseline`
- `Prefer baseline before reuse`

**Implemented:**
- Added one bounded replace-path decision on the existing inspect/detail decision seam for clearly regressed candidate runs.
- Strongly worse comparable evidence now switches the operator cue from `Compare regressed run` to `Keep baseline`.
- Reused the existing next-action seam so the replace path stays compact as `Prefer baseline before reuse`, without adding approvals, new persistence, or a separate review workflow.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py`

### Slice 5 — Validation freshness cue

**Status:** implemented

**Intent:** Dodać jeden bounded sygnał, czy ostatnie evidence jest nadal świeże operacyjnie.

**Good shape:**
- only if existing timestamps/history already support it,
- compact freshness wording,
- no scheduler or background policy.

**Implemented:**
- Reused the existing inspect/detail run-summary seam and appended one compact freshness cue derived only from the latest run `executed_at` timestamp.
- Recent runs now surface `Validation freshness: recent`, while older runs surface `Validation freshness: stale`.
- Kept the cue absent when execution timestamps are missing, without adding background refresh logic, new persistence, or a separate validation panel.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py`

### Slice 6 — CLI parity for evaluation cues

**Status:** implemented

**Intent:** Gdy Stage A-C dojrzeje, przenieść najważniejsze bounded semantics na headless surfaces.

**Good shape:**
- test-first parity assertions,
- no new CLI-only abstraction,
- shared product wording.

**Implemented:**
- Extended the existing `history-analytics` CLI summary so per-prompt rows can now surface the same bounded evaluation cues already visible in inspect/detail.
- Headless output now appends compact `decision`, `next`, and `freshness` lines when those shared summaries are already available, without adding a CLI-only scoring model, separate analytics workflow, or divergent wording.
- Reused the existing execution-analytics payload by adding optional bounded cue fields on `PromptExecutionAnalytics`, keeping GUI/CLI semantics aligned through one shared data seam.

**Verified:**
- `.venv/bin/pytest tests/test_main_entry.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py -q`
- `.venv/bin/ruff check cli/commands.py core/history_tracker.py tests/test_main_entry.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py`

### Slice 7 — Limited-evidence provenance cue

**Status:** implemented

**Intent:** Dodać jeden bounded ślad skąd bierze się fallback decyzji, gdy inspect/detail ma tylko cienkie run evidence zamiast porównywalnej pary albo lineage signal.

**Good shape:**
- provenance stays on the existing decision-support seam,
- only appears when thin run evidence already drives the fallback guidance,
- no new review workflow, persistence, or panel.

**Implemented:**
- Extended the existing decision provenance seam so inspect/detail now surfaces `Decision based on limited run evidence` whenever the decision falls back because history exists but comparable evidence is still too weak.
- Kept stronger provenance unchanged for comparable pairs (`latest 2 comparable runs`) and lineage-derived decisions (`fork lineage only`), so the new cue only tightens the thin-evidence path.
- Reused the shared detail/template-detail rendering path without adding any new widget state or analytics abstraction.

**Verified:**
- `.venv/bin/pytest tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py tests/test_main_entry.py -q`
- `.venv/bin/ruff check gui/workspace_history_controller.py tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py tests/test_main_entry.py cli/commands.py core/history_tracker.py`

---

## Selection rules for the first implementation slice

Pierwszy slice nowego cyklu powinien spełniać wszystkie warunki:
- reuse existing run/history/detail seams,
- adds one compact operator-facing semantic,
- has a clear RED test path,
- does not require schema/storage migration,
- does not require new screen/panel,
- strengthens evaluation/governance rather than feature breadth.

Jeżeli pierwszy kandydat okaże się już pokryty przez istniejące zachowanie, traktować to jako test-only guard albo przejść do kolejnego slice zamiast wymuszać zmianę runtime.

---

## Anti-goals for this cycle

W tym cyklu nie robić jako domyślnego kierunku:
- benchmark dashboardów,
- dużego score engine,
- nowego review queue,
- collaborative approvals,
- rozbudowanego workflow buildera,
- agent-style orchestration,
- nowego persistence modelu tylko dla evaluation,
- broad analytics-first UI.

---

## Documentation rule for this roadmap

Po każdej implementacji w tym cyklu:
1. najpierw zaktualizować ten roadmap,
2. dopisać krótki implemented/verified note,
3. zaktualizować `docs/CHANGELOG.md`,
4. aktualizować `docs/product-ssot.md` tylko jeśli produktowy priorytet albo definicja warstwy realnie się zmienia.

SSOT ma pozostać stabilny; roadmap ma być żywym ledgerem wykonania.

---

## Definition of done for the cycle

Ten cykl można uznać za wykonany w swoim planowanym zakresie. Kolejny execution boundary jest prowadzony w `docs/plans/2026-04-26-next-cycle-closure-and-next-plan.md`, żeby nie doklejać dalszych asset-to-run-to-refine slices do zamkniętego ledgeru evaluation/governance.

Ten cykl będzie można uznać za dobrze domknięty, gdy PromptManager będzie już nie tylko pokazywał bounded compare cues, ale też:
- lepiej komunikuje, czy evidence jest wystarczające,
- pokazuje bardziej czytelny governance/review posture,
- utrzymuje parity między głównymi surfaces,
- robi to bez utraty asset-first identity i bez budowy ciężkiej warstwy analytics/governance.
