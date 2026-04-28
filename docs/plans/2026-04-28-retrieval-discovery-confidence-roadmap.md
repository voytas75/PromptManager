# PromptManager Retrieval/Discovery Confidence Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć kolejny execution cycle po legibility v1 tak, aby operator nie tylko widział wynik retrieval, ale szybciej oceniał jego trafność i bezpieczniej przechodził z search do inspect/reuse bez zmiany rankingu ani otwierania nowego workflow.

**Architecture:** Ten roadmap pozostaje asset-first i zachowuje bounded retrieval/discovery posture. Zamiast nowego dashboardu lub explainability layer, wzmacnia istniejące seams w `PromptListModel`, `PromptListDelegate`, `PromptListCoordinator`, `PromptListPresenter` i ich testach, przesuwając fokus z samej legibility ku confidence cues: bardziej czytelny powód trafienia, spokojniejsze trust cues dla search state oraz nieco mocniejszą ciągłość search -> inspect.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, existing prompt list / preview / search flows, PromptManager product SSOT, docs under `docs/`.

---

## Why this roadmap exists

Live repo/docs/code review confirms:
- `docs/product-ssot.md` remains canonical and still keeps retrieval/discovery inside **Priority 1 — Prompt-asset core quality**,
- `docs/plans/2026-04-27-retrieval-discovery-legibility-roadmap.md` is practically closed: slices 1–4 are implemented, slice 5 is `CLI unchanged by design`, and slice 6 is `covered by existing behavior`,
- current repo state is clean, so the previous cycle has a clear execution boundary,
- existing retrieval/discovery v1 shipped only the narrowest explanation/trust cues:
  - `Matched in source` via `PromptListModel.MatchReasonRole`,
  - `Browsing all prompts` / `Showing search results` / `No matches for search` / `Search unavailable` via `PromptLoadResult.operator_state_label`,
  - the similar-result presenter status cue with a compact inspect handoff,
- current code/tests do **not** yet show a broader confidence layer for equally-bounded title/scenario matches or for slightly richer operator-facing retrieval trust wording.

So the next useful move is not another SSOT rewrite and not a new product-definition cycle, but a fresh execution ledger that extends retrieval/discovery from **legibility v1** into **confidence v1**.

Short version:

**asset-first core -> retrieval/discovery legibility -> retrieval/discovery confidence**

---

## Confirmed baseline before this cycle

Assume already delivered and do not reopen without a failing regression:
- bounded retrieval previews in the main prompt list,
- active-search highlight spans for title and preview text,
- bounded source-match reason cue: `Matched in source`,
- coordinator-local search-state posture labels,
- similar-result recommendation wording plus bounded inspect handoff hint,
- GUI-local parity decision for retrieval cues,
- guard-pack coverage for prompt-list preview/match/state behavior.

This cycle should build on those seams instead of replacing them.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager już komunikuje podstawową legibility retrieval, to jak sprawić, żeby operator szybciej ufał wynikom search i łatwiej oceniał, *czy ten wynik wygląda trafnie i co z nim zrobić dalej*, nadal bez ciężkiego explainability workflow?

Nowy cykl wzmacnia trzy rzeczy:

1. **match-confidence clarity** — operator ma szybciej rozumieć, *w czym* wynik pasuje,
2. **retrieval trust wording** — search state ma być trochę bardziej operacyjny, ale nadal bounded,
3. **search-to-inspect continuity** — search results mają spokojniej sugerować właściwy następny ruch bez dublowania inspect/detail logic.

---

## Constraints

Do not add:
- a retrieval dashboard,
- score-heavy ranking explanation UI,
- a search analytics surface,
- a new persistence model for explanation/confidence cues,
- hidden ranking changes,
- CLI/headless expansion unless a cue becomes shared product truth,
- a second recommendation surface that competes with inspect/detail.

Reuse first:
- `gui/prompt_list_model.py`
- `gui/prompt_list_delegate.py`
- `gui/prompt_list_coordinator.py`
- `gui/prompt_list_presenter.py`
- existing preview/match helpers
- focused prompt-list tests already present in `tests/`

---

## Roadmap stages for this cycle

### Stage A — Broaden bounded match-reason coverage

Cel: wyjść poza sam `Matched in source`, ale nadal pozostać przy jednym małym explanation seam.

Obszary:
- `Matched in title` for visible title-match-first paths,
- `Matched in scenario` when preview was promoted from a matching scenario,
- no new ranking logic.

Constraint:
- expand only where existing visible text already justifies the cue,
- do not invent semantic-heavy wording in v1,
- prefer model-role and delegate parity over presenter/state changes.

### Stage B — Refine retrieval-state confidence wording

Cel: zrobić search-state posture trochę bardziej actionable i spokojne, bez otwierania nowej warstwy statusów.

Obszary:
- wording polish around active search and no-match states,
- bounded distinction between ordinary search posture and recommendation posture,
- no collapse of distinct states.

Constraint:
- keep the seam coordinator-local first,
- avoid turning state labels into a second narrative that duplicates list content.

### Stage C — Strengthen search-to-inspect continuity

Cel: lepiej podpowiedzieć następny krok z wyników search, ale bez kopiowania inspect/detail decision cues do listy.

Obszary:
- compact handoff wording on search/similar-result seams,
- maybe one slightly stronger inspect-oriented phrase if tests/runtime show it stays calm,
- continuity without workflow expansion.

Constraint:
- inspect/detail remains canonical for actual decisions,
- list-level cues must stay lightweight and subordinate.

### Stage D — Guard and locality discipline

Cel: po 1–2 runtime slices domknąć parity/reset/no-op coverage i jasno potwierdzić, co pozostaje GUI-local.

Obszary:
- inactive-search reset guards,
- cue absence on weak/non-credible paths,
- locality-vs-shared checks,
- no-op closures where runtime already behaves correctly.

Constraint:
- prefer test-only closure when runtime seams already do the right thing.

---

## Recommended execution order

1. broaden match-reason cues on existing visible text
2. refine search-state trust wording if a real bounded gap remains
3. strengthen search-to-inspect continuity wording
4. close parity/locality/guard pack for the new cues

---

## Candidate bounded slices

### Slice 1 — Match reason expansion v1 (`title` / `scenario`)

**Status:** implemented

**Intent:** Rozszerzyć istniejący `MatchReasonRole` tak, aby aktywny search mógł pokazać nie tylko `Matched in source`, ale także bounded reason cue dla widocznego matchu w title albo promoted scenario preview, nadal bez zmiany rankingu.

**Why this next:**
- directly extends the existing v1 seam instead of inventing a new one,
- fits the current README/SSOT focus on retrieval/discovery clarity,
- stays inside `PromptListModel` + delegate/tests,
- is the clearest next step from legibility toward confidence.

**Implemented:**
- `PromptListModel.MatchReasonRole` now returns `Matched in scenario` when the active-search preview is a promoted matching scenario preview already visible in the row,
- the same seam now returns `Matched in title` when the visible title is the bounded active-search match path and no stronger preview-specific cue wins,
- existing `Matched in source` behavior stays intact for credible source-promoted previews,
- ranking, persistence, coordinator state labels, and presenter workflow remain unchanged.

**Good shape:**
- `Matched in title` when the visible title carries the relevant bounded match and no stronger preview-specific cue should win,
- `Matched in scenario` when active search promotes a scenario preview,
- preserve current `Matched in source` behavior for source-promoted previews,
- keep cue absent for no-search, weak preview, or ordinary no-cue paths.

**Likely files:**
- Modify: `gui/prompt_list_model.py`
- Maybe modify: `gui/prompt_list_delegate.py` only if rendering needs a tiny follow-up parity adjustment
- Modify: `tests/test_prompt_list_model.py`

**Verification target:**
- `.venv/bin/pytest tests/test_prompt_list_model.py -q`
- `.venv/bin/pytest tests/test_prompt_list_model.py tests/test_prompt_list_coordinator.py -q`
- `.venv/bin/ruff check gui/prompt_list_model.py tests/test_prompt_list_model.py`

**Verification:**
- RED confirmed first with `.venv/bin/pytest tests/test_prompt_list_model.py::test_prompt_list_model_prefers_matching_scenario_over_non_matching_description tests/test_prompt_list_model.py::test_prompt_list_model_reports_title_match_reason_when_title_is_visible_match -q` -> `2 failed` on missing `MatchReasonRole` cues,
- GREEN after the minimal model-only change with the same targeted pytest command -> `2 passed`,
- broader slice verification: `.venv/bin/pytest tests/test_prompt_list_model.py -q` -> `16 passed`,
- nearby smoke: `.venv/bin/pytest tests/test_prompt_list_model.py tests/test_prompt_list_coordinator.py -q` -> `23 passed`,
- lint: `.venv/bin/ruff check gui/prompt_list_model.py tests/test_prompt_list_model.py` -> `All checks passed!`,
- locality verdict: GUI-local only; no shared analytics or CLI/headless expansion needed.

### Slice 2 — Search-state confidence wording polish

**Status:** pending after Slice 1

**Intent:** Sprawdzić, czy obecne state labels są już wystarczające, czy potrzebują jednego bounded wording pass, żeby active search / no-match posture brzmiały trochę bardziej action-oriented i trustable.

**Good shape:**
- keep the current four-state contract,
- allow wording polish only if runtime/tests show a real ambiguity,
- preserve `Search unavailable` as explicit error posture.

**Likely files:**
- Modify: `gui/prompt_list_coordinator.py`
- Maybe modify: nearby presenter/status consumer only if the cue is already surfaced there
- Modify: `tests/test_prompt_list_coordinator.py`

**Decision rule:**
- if the first RED test passes immediately and current wording already feels sufficiently explicit, close this slice as `covered by existing behavior`.

### Slice 3 — Search-to-inspect continuity polish

**Status:** pending after Slice 2

**Intent:** Zobaczyć, czy ordinary search path — nie tylko similar-result mode — potrzebuje jednego bounded inspect handoff hint, żeby przejście z search do detail było bardziej oczywiste.

**Good shape:**
- wording-only first,
- no new panel,
- no duplication of `Decision` / `Recommended next action` cues from inspect/detail.

**Likely files:**
- Modify: `gui/prompt_list_presenter.py`
- Add/modify: focused presenter tests

**Decision rule:**
- only land this if a compact wording seam exists and stays calmer than a second recommendation layer.

### Slice 4 — Confidence-cue locality/parity guard pack

**Status:** pending after runtime slices

**Intent:** Domknąć nowy confidence layer lekkimi testami i decyzją shared-vs-local.

**Good shape:**
- add guard coverage for no-search / weak-preview / reset cases,
- verify new retrieval confidence cues remain GUI-local unless they truly enter shared analytics,
- avoid runtime churn created only to satisfy plan bookkeeping.

**Likely files:**
- Modify: `tests/test_prompt_list_model.py`
- Maybe modify: `tests/test_retrieval_cues_parity.py`
- Maybe modify: roadmap docs only

---

## Recommended first slice

### Pick first: Slice 1 — Match reason expansion v1 (`title` / `scenario`)

Why this first:
- it is the smallest real product extension still clearly justified by SSOT,
- current code already has the `MatchReasonRole` seam,
- current tests already cover source/scenario/title matching behavior nearby,
- it should produce a user-visible confidence gain without changing ranking, persistence, or workflow.

Why not start with roadmap-wide wording churn:
- state wording and continuity wording are more subjective,
- the next strongest confidence gain comes from making the existing reason seam slightly broader first,
- a model-local slice is cheaper to verify than presenter/coordinator copy churn.

---

## Implementation brief for the first slice

### Task 1: Confirm current match-reason priority behavior

**Objective:** Read the current `PromptListModel` reason-cue logic and the nearest active-search tests before changing assertions.

**Files:**
- Read: `gui/prompt_list_model.py`
- Read: `tests/test_prompt_list_model.py`
- Maybe read: `gui/prompt_list_delegate.py`

**Verification:**
- confirm current `MatchReasonRole` only returns `Matched in source`,
- confirm the nearest safe scenario/title test seams already exist.

### Task 2: Write failing tests for title/scenario reason cues

**Objective:** Prove the current bounded reason seam does not yet expose the broader confidence cues.

**Files:**
- Modify: `tests/test_prompt_list_model.py`

**Verification:**
- `.venv/bin/pytest tests/test_prompt_list_model.py -q`
- Expected before implementation: failing assertions on new `MatchReasonRole` expectations.

### Task 3: Implement the minimal `MatchReasonRole` expansion only

**Objective:** Add the smallest bounded logic that emits `Matched in title` / `Matched in scenario` where the currently visible row content already justifies it.

**Files:**
- Modify: `gui/prompt_list_model.py`
- Maybe modify: `gui/prompt_list_delegate.py` only if needed for rendering parity

**Verification:**
- `.venv/bin/pytest tests/test_prompt_list_model.py -q`
- `.venv/bin/pytest tests/test_prompt_list_model.py tests/test_prompt_list_coordinator.py -q`
- `.venv/bin/ruff check gui/prompt_list_model.py tests/test_prompt_list_model.py`

### Task 4: If RED passes immediately, convert to guard-only closure

**Objective:** Avoid fake runtime churn if existing seams already cover the broader reason behavior.

**Files:**
- Modify: this roadmap
- Maybe modify: `docs/CHANGELOG.md` only if the new guard materially matters for repo history

**Verification:**
- record exact pytest command/result,
- mark the slice as `covered by existing behavior` or `implemented` based on the real outcome.

### Task 5: Re-apply locality rule

**Objective:** Confirm whether the broader confidence cue remains GUI-local.

**Files:**
- Maybe modify: `tests/test_retrieval_cues_parity.py`
- Only touch CLI/headless seams if the cue escapes the prompt-list surface

**Verification:**
- document `CLI unchanged by design` unless the cue genuinely enters shared analytics.

### Task 6: Sync ledger after green

**Objective:** Keep this roadmap as the canonical execution ledger for the new retrieval/discovery confidence cycle.

**Files:**
- Modify: this file
- Modify: `docs/CHANGELOG.md` if user-visible behavior changed
- Update `docs/product-ssot.md` only if product posture or priority order changed

---

## Documentation rule for this roadmap

Po każdej implementacji w tym cyklu:
1. najpierw zaktualizować ten roadmap,
2. dopisać krótki implemented / verified note,
3. zaktualizować `docs/CHANGELOG.md` jeśli zmieniło się user-visible behavior albo execution ledger wymaga krótkiego śladu,
4. aktualizować `docs/product-ssot.md` tylko jeśli zmienia się definicja produktu albo porządek priorytetów.

SSOT ma pozostać stabilny; ten plik ma być żywym execution ledgerem.

---

## Definition of done for this cycle

Ten cykl będzie można uznać za dobrze domknięty, gdy PromptManager:
- lepiej komunikuje *w czym* wynik retrieval pasuje,
- utrzymuje bounded i spokojne trust cues dla search states,
- wzmacnia przejście z search do inspect bez ciężkiego workflow,
- zachowuje asset-first posture,
- i nadal jasno rozróżnia, które retrieval cues są shared truth, a które pozostają GUI-local by design.
