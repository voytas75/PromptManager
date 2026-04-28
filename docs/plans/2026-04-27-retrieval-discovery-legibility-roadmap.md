# PromptManager Retrieval/Discovery Legibility Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć nowy execution cycle skupiony na retrieval/discovery clarity tak, aby operator szybciej rozumiał, dlaczego dany prompt wynik pojawił się na liście i jak bezpiecznie przejść z retrieval do inspect/reuse.

**Architecture:** Ten roadmap nie zmienia asset-first modelu produktu ani nie otwiera nowego retrieval workflow. Zamiast tego wzmacnia istniejące seams w `PromptListModel`, `PromptListDelegate`, `PromptListCoordinator`, `PromptListPresenter` i powiązanych testach, dodając bounded explanation cues, lepszą legibility search state i ostrożną ciągłość między listą wyników a inspect entry path.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, existing prompt list / preview / search flows, PromptManager product SSOT, docs under `docs/`.

---

## Why this roadmap exists

Live repo/docs review confirms:
- `docs/product-ssot.md` remains the canonical SSOT and still centers PromptManager on prompt assets, retrieval, inspect, reuse, and refinement,
- `docs/plans/2026-04-26-asset-to-run-to-refine-coherence-v3-roadmap.md` is practically exhausted and no longer a good active execution ledger,
- `docs/plans/2026-04-27-retrieval-discovery-roadmap-prep.md` already records the product rationale for moving to retrieval/discovery leverage,
- README now explicitly signals retrieval/discovery clarity as part of the current focus,
- the existing codebase already has reusable seams for prompt-list preview choice, search highlighting, and retrieval-state handling,
- the nearest focused tests live in `tests/test_prompt_list_model.py` and `tests/test_prompt_list_coordinator.py`.

So the next useful move is not another inspect/workspace micro-slice, but a fresh ledger that sharpens retrieval/discovery legibility inside the existing prompt-list surfaces.

Short version:

**asset-first core -> trustworthy runs -> inspect/workspace coherence -> retrieval/discovery legibility**

---

## Confirmed baseline before this cycle

Assume already delivered:
- bounded retrieval previews in the main prompt list,
- active-search match highlighting for visible title and preview text,
- source/scenario/body fallback logic for preview choice,
- search-state distinction between `no matches` and `search error`,
- favorites-only filtering and existing prompt-list sorting/filter seams,
- semantic-similar prompt entry path via the existing presenter/controller flow.

This cycle should not re-open those foundations unless a failing regression proves they are insufficient for the next bounded cue.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager umie już znaleźć i pokazać prompty, to jak zrobić, żeby operator szybciej ufał wynikom retrieval i sprawniej przechodził z listy do właściwego reuse/inspect path bez nowego ciężkiego workflow?

Nowy cykl wzmacnia trzy rzeczy:

1. **search-result legibility** — wynik ma być łatwiejszy do zrozumienia bez zgadywania, dlaczego się pojawił,
2. **retrieval trust cues** — search i similar-result paths mają lepiej komunikować swój bounded meaning,
3. **retrieval-to-inspect continuity** — przejście z listy wyników do dalszej akcji ma być czytelniejsze, ale nadal lekkie.

---

## Constraints

Do not add:
- a new retrieval dashboard,
- a search analytics panel,
- a separate ranking explanation screen,
- a new persistence model only for explanation cues,
- hidden ranking changes without visible rationale,
- broad workflow expansion around review queues or collections,
- CLI/headless growth unless a cue becomes shared product truth first.

Reuse first:
- `gui/prompt_list_model.py`
- `gui/prompt_list_delegate.py`
- `gui/prompt_list_coordinator.py`
- `gui/prompt_list_presenter.py`
- existing prompt-preview helpers
- existing tests around prompt-list model and coordinator behavior

---

## Roadmap stages for this cycle

### Stage A — Search-result reason clarity

Cel: sprawić, żeby użytkownik szybciej rozumiał, dlaczego wynik jest trafny.

Obszary:
- bounded match-reason cues,
- search-result rationale visible on the existing list seam,
- no ranking changes in the first pass.

Constraint:
- start with one compact explanation seam,
- prefer text derived from already-known visible matches before adding semantic-specific wording.

### Stage B — Retrieval-state trust clarity

Cel: doprecyzować różnicę między typami retrieval state, tak aby no-match, search-error, active-search i similar-result paths były czytelniejsze operacyjnie.

Obszary:
- list-state wording,
- status / trust posture around search vs recommendations,
- bounded operator cues without new panels.

Constraint:
- preserve existing state contracts,
- do not collapse distinct search states back into one generic empty state.

### Stage C — Retrieval-to-inspect continuity

Cel: poprawić ciągłość między listą wyników a następną sensowną akcją bez rozbudowy detail workflow.

Obszary:
- inspect-entry clarity,
- list-to-detail continuity cues,
- bounded similar-result semantics.

Constraint:
- keep inspect/detail as the canonical decision surface,
- avoid duplicating inspect guidance directly in the list unless it stays compact and non-competing.

### Stage D — Guard and parity discipline

Cel: po 1–2 runtime slices domknąć najbliższe ryzyka lekkimi guardami zamiast poszerzać feature set.

Obszary:
- preview-choice stability,
- rationale-cue reset behavior,
- state-specific no-op guards,
- shared-vs-local parity decisions where needed.

Constraint:
- test-only closure is preferred over forced runtime churn when the seam already behaves correctly.

---

## Recommended execution order

1. search match reason clarity
2. retrieval-state trust wording / status clarity
3. similar-result or inspect-entry continuity refinement
4. shared-semantics parity or explicit no-op
5. test-only guard slices around prompt-list behavior

---

## Candidate bounded slices

### Slice 1 — Search match reason cue v1

**Status:** implemented

**Intent:** Dodać jeden bounded explanation cue przy aktywnym search path, tak aby operator widział, czemu dany wynik jest trafny, bez zmiany rankingu ani layout explosion.

**Good shape:**
- reuse `PromptListModel` / `PromptListDelegate` seams,
- derive wording from already-detected match source when possible,
- keep one compact visible cue,
- no new persistence,
- no ranking rewrite.

**Examples:**
- `Matched in title`
- `Matched in source`
- `Matched in scenario`
- start narrower if runtime/tests show only one safe first seam.

**Implemented:**
- Added one bounded `PromptListModel.MatchReasonRole` seam for active-search result rows.
- The first shipped cue stays intentionally narrow: when active search promotes a credible source preview (`Source: ...`), the same row can now expose `Matched in source` as a compact reason cue.
- Kept the slice model-local and ranking-neutral: no preview-order rewrite, no new persistence, no presenter state change, and no CLI/headless parity expansion.

**Likely files:**
- Modify: `gui/prompt_list_model.py`
- Maybe modify: `gui/prompt_list_delegate.py`
- Modify: `tests/test_prompt_list_model.py`

**Verified:**
- `.venv/bin/pytest tests/test_prompt_list_model.py::test_prompt_list_model_prefers_matching_source_preview_for_active_search -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_list_model.py -q`
- result: `15 passed`
- `.venv/bin/pytest tests/test_prompt_list_model.py tests/test_prompt_list_coordinator.py -q`
- result: `19 passed`
- `.venv/bin/ruff check gui/prompt_list_model.py tests/test_prompt_list_model.py`
- result: `All checks passed!`

### Slice 2 — Search-state operator trust cue

**Status:** implemented

**Intent:** Uczytelnić operacyjnie, czy użytkownik patrzy na no-match search state, search error, czy ordinary catalog state, bez rozbijania istniejących kontraktów.

**Good shape:**
- reuse `PromptListCoordinator` and existing presenter status handling,
- clarify wording/state handling only where current behavior feels too implicit,
- keep distinction between `no matches` and `search error` explicit.

**Implemented:**
- Extended `PromptLoadResult` with one bounded trust/state cue seam: `operator_state_label`.
- The current v1 state labels are intentionally compact and coordinator-local:
  - `Browsing all prompts` for the ordinary catalog posture,
  - `Showing search results` for active search with matches,
  - `No matches for search` for active search with zero results,
  - `Search unavailable` when the search backend raises an error.
- Kept the slice contract-safe: no ranking changes, no presenter workflow rewrite, no new persistence, and no CLI/headless parity expansion.

**Likely files:**
- Modify: `gui/prompt_list_coordinator.py`
- Maybe modify: `gui/prompt_list_presenter.py`
- Modify: `tests/test_prompt_list_coordinator.py`

**Verified:**
- `.venv/bin/pytest tests/test_prompt_list_coordinator.py::test_fetch_prompts_exposes_operator_state_for_no_match_search -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_list_coordinator.py::test_fetch_prompts_exposes_operator_state_for_search_errors -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_list_coordinator.py::test_fetch_prompts_exposes_operator_state_for_default_catalog -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_list_coordinator.py -q`
- result: `7 passed`
- `.venv/bin/pytest tests/test_prompt_list_model.py tests/test_prompt_list_coordinator.py -q`
- result: `22 passed`
- `.venv/bin/ruff check gui/prompt_list_coordinator.py tests/test_prompt_list_coordinator.py`
- result: `All checks passed!`

### Slice 3 — Similar-result path meaning clarity

**Status:** implemented

**Intent:** Jeżeli operator wchodzi w podobne prompty, lista powinna lepiej sygnalizować, że to recommendation/similarity path, a nie zwykły text search result.

**Good shape:**
- reuse existing `show_similar_prompts()` path,
- prefer one bounded status or visible state cue,
- no new recommendation screen.

**Implemented:**
- Tightened the existing `PromptListPresenter.show_similar_prompts()` status seam so similar-result mode now announces itself as a recommendation path instead of looking like ordinary search output.
- The shipped v1 cue is intentionally compact and list-local: `Showing similar prompts for '<name>'. Recommendation results only.`
- Kept the slice minimal: no new screen, no ranking rewrite, no persistence changes, and no CLI/headless parity expansion.

**Likely files:**
- Modify: `gui/prompt_list_presenter.py`
- Maybe modify: prompt-list view status seam already used by callbacks
- Add/modify focused presenter or coordinator tests if available; otherwise start with the smallest nearby seam test

**Verified:**
- `.venv/bin/pytest tests/test_prompt_list_presenter.py::test_show_similar_prompts_surfaces_recommendation_state_cue -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_list_presenter.py -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q`
- result: `12 passed`
- `.venv/bin/ruff check gui/prompt_list_presenter.py tests/test_prompt_list_presenter.py`
- result: `All checks passed!`

### Slice 4 — Retrieval-to-inspect continuity cue

**Status:** implemented

**Intent:** Sprawdzić, czy list-level retrieval clue może lekko wspierać przejście do inspect bez dublowania decision-support cues z detail view.

**Good shape:**
- stay compact,
- no inspect-logic duplication,
- no new panels,
- likely wording-only or guard-first.

**Implemented:**
- Extended the existing similar-result list status seam with one bounded handoff hint so recommendation mode now points operators toward inspect as the next safe step.
- The shipped continuity cue remains intentionally compact and local to the presenter status message: `inspect a prompt for reuse details`.
- Kept the slice wording-only: no inspect logic duplication, no new panels, no ranking/persistence changes, and no CLI/headless parity expansion.

**Verified:**
- `.venv/bin/pytest tests/test_prompt_list_presenter.py::test_show_similar_prompts_adds_bounded_inspect_handoff_cue -q`
- result: `1 passed`
- `.venv/bin/pytest tests/test_prompt_list_presenter.py -q`
- result: `2 passed`
- `.venv/bin/pytest tests/test_main_window_bridges.py tests/test_template_preview_widget.py -q`
- result: `12 passed`
- `.venv/bin/ruff check gui/prompt_list_presenter.py tests/test_prompt_list_presenter.py`
- result: `All checks passed!`

### Slice 5 — Headless/shared parity decision for retrieval cues

**Status:** CLI unchanged by design

**Intent:** Każdy nowy cue z tego cyklu musi przejść przez shared-vs-local rule.

**Decision rule:**
- if the cue remains list-local and presentation-specific, CLI stays unchanged by design,
- if a cue becomes shared retrieval truth beyond GUI rendering, inspect the existing headless seam before adding parity.

**Decision:**
- The current retrieval/discovery cues from this cycle remain GUI-local by design:
  - `PromptListModel.MatchReasonRole`
  - `PromptLoadResult.operator_state_label`
  - the similar-result presenter status wording / inspect handoff hint
- None of them currently travel through shared analytics fields such as `decision_summary`, `next_action_summary`, or `freshness_summary`, and none are consumed by the existing headless `history-analytics` seam.
- Therefore CLI/headless behavior stays intentionally unchanged for this cycle.

**Implemented:**
- Added one focused parity guard test to lock the current decision boundary: prompt-list retrieval cues exist, but they are not shared analytics summaries.
- Kept the slice docs/test-only instead of inventing shadow CLI wording or expanding shared models.

**Verified:**
- `.venv/bin/pytest tests/test_retrieval_cues_parity.py -q`
- result: `1 passed`

### Slice 6 — Prompt-list guard pack

**Status:** covered by existing behavior

**Intent:** Po runtime slice’ach domknąć regresje lekkimi testami zamiast rozpychać feature scope.

**Decision:**
- A focused doc/code/test audit shows the nearest guard pack is already materially covered by the current prompt-list test set, so this slice should close as a ledger sync instead of inventing fake runtime churn.
- Existing coverage already locks the intended guard categories:
  - reset / inert behavior when search is blank: `test_prompt_list_model_keeps_no_search_preview_priority_unchanged`
  - no-op match roles without active search: `test_prompt_list_model_keeps_match_roles_empty_without_active_search`
  - cue/preview parity around active-search preview selection and match spans: `test_prompt_list_model_prefers_matching_source_preview_for_active_search`, `test_prompt_list_model_prefers_matching_scenario_over_non_matching_description`, `test_prompt_list_model_exposes_title_and_preview_match_spans_for_active_search`
  - state-specific negative guards for retrieval trust posture: `test_fetch_prompts_exposes_operator_state_for_no_match_search`, `test_fetch_prompts_exposes_operator_state_for_search_errors`, `test_fetch_prompts_exposes_operator_state_for_default_catalog`
- No additional runtime seam, presenter churn, or CLI/headless expansion is needed for closure.

**Good shape:**
- reset on search clear,
- no-op on inactive search,
- cue/preview parity with existing highlight paths,
- state-specific negative guards.

**Covered by existing behavior:**
- The current prompt-list regression set already covers the intended guard pack without needing a new implementation slice.
- Closure is documentation-only: treat this as a stale pending marker reconciled to the present tested behavior.

**Verified:**
- `.venv/bin/pytest tests/test_prompt_list_model.py tests/test_prompt_list_coordinator.py tests/test_prompt_list_presenter.py tests/test_retrieval_cues_parity.py -q`
- do weryfikacji live run in this session
- expected/last known scope from the current implemented slices: prompt-list guards remain covered by the focused retrieval/discovery test set.

---

## Recommended first slice

### Pick first: Slice 1 — Search match reason cue v1

Why this first:
- directly targets the new roadmap’s core question,
- stays inside proven prompt-list seams,
- can likely land as one bounded user-visible cue,
- does not require new storage or workflow,
- should quickly reveal whether the best first move is runtime wording or guard-only coverage.

Why not start with state trust first:
- the state-contract groundwork already exists,
- search-result reason clarity is the more central user-facing leverage point,
- state trust wording can follow once the primary cue seam is understood.

---

## Implementation brief for the first slice

### Task 1: Confirm current prompt-list search cue behavior

**Objective:** Inspect the exact current model/delegate behavior for active-search result rows before writing assertions.

**Files:**
- Read: `gui/prompt_list_model.py`
- Read: `gui/prompt_list_delegate.py`
- Read: `tests/test_prompt_list_model.py`

**Verification:**
- identify whether any existing role already exposes an explanation-friendly seam beyond preview/highlight,
- confirm the smallest likely test seam for a new bounded cue.

### Task 2: Write failing test for the first match-reason cue

**Objective:** Prove the current prompt-list seam does not yet surface the desired reason clarity.

**Files:**
- Modify: `tests/test_prompt_list_model.py`
- Maybe modify: delegate tests only if rendering needs a visible label beyond model role assertions

**Verification:**
- `.venv/bin/pytest tests/test_prompt_list_model.py -q`
- Expected before implementation: one failing assertion on the new reason-cue role or visible output

### Task 3: Implement minimal model/delegate change only

**Objective:** Add the smallest bounded cue that makes the failing test pass without changing search ranking.

**Files:**
- Modify: `gui/prompt_list_model.py`
- Maybe modify: `gui/prompt_list_delegate.py`

**Verification:**
- `.venv/bin/pytest tests/test_prompt_list_model.py -q`
- broader nearby smoke after green

### Task 4: If RED passes immediately, convert the slice into a guard-only closure

**Objective:** Avoid fake runtime churn if the current seam already gives enough reason clarity.

**Files:**
- Modify: this roadmap
- Maybe modify: `docs/CHANGELOG.md` only if the test-only guard materially strengthens the execution ledger

**Verification:**
- record exact pytest command/result,
- mark the slice as `covered by existing behavior` or `implemented` depending on whether new regression coverage landed.

### Task 5: Decide on parity / locality

**Objective:** Apply the shared-vs-local rule to any retrieval cue that survives the slice.

**Files:**
- Maybe modify: `tests/test_main_entry.py`
- Maybe modify: other shared seams only if the cue stops being GUI-local

**Verification:**
- `.venv/bin/pytest tests/test_main_entry.py -q` if touched,
- otherwise document `CLI unchanged by design`.

### Task 6: Sync roadmap ledger after green

**Objective:** Keep this roadmap as the canonical execution ledger for the retrieval/discovery cycle.

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
4. aktualizować `docs/product-ssot.md` tylko jeśli zmieni się definicja produktu albo porządek priorytetów.

SSOT ma pozostać stabilny; ten plik ma być żywym ledgerem wykonania.

---

## Definition of done for this cycle

Ten cykl będzie można uznać za dobrze domknięty, gdy PromptManager:
- lepiej komunikuje, dlaczego wynik retrieval pojawił się na liście,
- utrzymuje czytelną różnicę między search states i recommendation paths,
- wzmacnia przejście z retrieval do inspect/reuse bez ciężkiego workflow,
- zachowuje asset-first posture,
- oraz ma jasną regułę, które retrieval semantics są shared, a które pozostają local by design.
