# PromptManager Prompt-Library Filter Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć kolejny execution cycle po retrieval/discovery confidence tak, aby istniejące filtry biblioteki promptów były bardziej czytelne operacyjnie i lepiej wspierały retrieve -> inspect -> reuse bez zmiany semantyki filtrowania, rankingu ani persistence.

**Architecture:** Ten roadmap zostaje w Stage 2 asset-loop quality i celowo używa tylko istniejących seams wokół `PromptFilterPanel` oraz najbliższych prompt-library consumers. Zamiast przebudowy panelu filtrów, wielotagowego workflow albo search redesignu, skupia się na bounded visibility cues, active-filter clarity i lekkich guardach reset/parity tam, gdzie operatorowi brakuje jawnej informacji o zawężeniu biblioteki.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, existing prompt-library filter panel and prompt list flows, PromptManager SSOT docs under `docs/`.

---

## Why this roadmap exists

Live repo/docs/code review confirms:
- `docs/product-ssot.md` nadal priorytetyzuje jakość core prompt asset loop,
- aktywny retrieval/discovery cycle w `docs/plans/2026-04-28-retrieval-discovery-confidence-roadmap.md` jest praktycznie domknięty: wszystkie slice’y są `implemented`, a search po `**Status:** pending` i `[ ]` w `docs/plans/` zwraca zero otwartych zadań,
- repo jest clean po ostatnich bounded context-menu clarity slices,
- PromptManager ma już istniejący filter-panel seam blisko głównego celu produktu: organizować i odzyskiwać prompt assets,
- w repo istnieje już pierwszy mały usability gain na tym seamie: `Tag filter visibility v1` został dowieziony lokalnie w `PromptFilterPanel`, ale nie ma jeszcze świeżego execution ledgeru dla całej rodziny filter-clarity slices.

So the next useful move is not another retrieval roadmap extension and not a broader SSOT rewrite, but a fresh bounded execution ledger focused on **prompt-library filter clarity**.

Short version:

**asset-first core -> retrieval/discovery confidence -> prompt-library filter clarity**

---

## Confirmed baseline before this cycle

Assume already delivered and do not reopen without a failing regression:
- bounded retrieval/discovery confidence cues in prompt-list seams,
- operator-facing context-menu clarity for `Execute as Context…`, `Fork Prompt`, `Similar Prompts`, `Duplicate Prompt`, and `Execute Prompt`,
- `Recent Prompts` metadata rows,
- `Show Description` empty-state wording improvement,
- `PromptFilterPanel` tag helper cue v1 with visible `Tag filter: all tags` / `Tag filter: <tag>` wording,
- no change in tag filter semantics, persistence, ranking, or prompt-list search flow.

This cycle should build on the existing filter-panel seam instead of replacing it.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager ma już działające filtry biblioteki promptów, to jak zrobić je bardziej czytelnymi i spokojniejszymi operacyjnie, żeby operator szybciej widział *jak biblioteka jest zawężona* i bezpieczniej przechodził do inspect/reuse, nadal bez przebudowy workflow filtrów?

Nowy cykl wzmacnia trzy rzeczy:

1. **active-filter visibility** — operator ma szybciej widzieć, że biblioteka jest zawężona,
2. **filter-state trust cues** — reset/default state i active state mają być bardziej jawne, ale nadal lekkie,
3. **filter-to-list continuity** — cues mają wspierać korzystanie z listy promptów bez tworzenia drugiego systemu statusów.

---

## Constraints

Do not add:
- multi-tag selection,
- tag counts or analytics badges,
- ranking/search scoring changes,
- new persistence for filter UI state,
- filter dashboard or advanced query builder,
- CLI/headless parity unless a cue becomes shared product truth,
- a second recommendation/status layer competing with the prompt list.

Reuse first:
- `gui/widgets/prompt_filter_panel.py`
- existing prompt-library consumers already wired to the panel
- focused filter-panel and prompt-list tests in `tests/`

---

## Roadmap stages for this cycle

### Stage A — Visible active-filter posture

Cel: sprawić, żeby aktywne zawężenie biblioteki było czytelniejsze bez zmiany logiki filtrowania.

Obszary:
- explicit active-tag visibility,
- calm all-tags default cue,
- maybe one nearby active-filter helper for another existing control only if a real bounded gap remains.

Constraint:
- stay inside the filter-panel seam first,
- no filter semantics change.

### Stage B — Reset/default-state clarity

Cel: upewnić się, że operator łatwo rozumie, kiedy panel wraca do stanu neutralnego.

Obszary:
- reset to all/default wording,
- guard coverage for programmatic repopulation,
- no extra reset workflow.

Constraint:
- prefer tests/guards if runtime already behaves correctly.

### Stage C — Filter-to-list continuity

Cel: delikatnie poprawić ciągłość między aktywnymi filtrami a czytaniem listy promptów.

Obszary:
- one compact continuity cue if an existing visible seam justifies it,
- avoid duplicating prompt-list search/retrieval statuses,
- keep list-level semantics unchanged.

Constraint:
- list remains canonical for results,
- filter cues stay subordinate.

### Stage D — Guard/locality closure

Cel: domknąć cycle lekkimi parity/reset/no-op checks i jasno oznaczyć, co pozostaje GUI-local.

Obszary:
- reset guard coverage,
- no-op closure if a candidate slice is already covered,
- locality decision for filter cues.

Constraint:
- prefer test-only closure when runtime already does the right thing.

---

## Recommended execution order

1. lock the shipped tag visibility v1 as the baseline of the new cycle
2. inspect whether reset/default clarity still has a real bounded gap
3. probe one small filter-to-list continuity cue only if it stays calmer than a second status layer
4. close guard/locality coverage

---

## Candidate bounded slices

### Slice 1 — Tag filter visibility v1 baseline

**Status:** implemented

**Intent:** Dodać w istniejącym `PromptFilterPanel` mały, zawsze widoczny cue pokazujący, czy biblioteka jest filtrowana po tagu, bez zmiany logiki filtrów.

**Implemented:**
- `PromptFilterPanel` now renders a compact helper label with `Tag filter: all tags` when no tag is selected,
- the same seam now shows `Tag filter: <tag>` for an active tag,
- user-driven tag changes update the helper cue before emitting `filters_changed`,
- programmatic `set_tags(...)` repopulation also refreshes the visible cue,
- filter semantics, persistence, ranking, category/favorites/quality behavior, and prompt-list retrieval flow remain unchanged.

**Likely files:**
- Modify: `gui/widgets/prompt_filter_panel.py`
- Add/modify: `tests/test_prompt_filter_panel.py`
- Modify: `docs/CHANGELOG.md`
- Brief already exists: `docs/implementation-brief-2026-04-28-tag-filter-visibility-v1.md`

**Verification baseline:**
- targeted filter-panel tests exist for calm default state, active tag state, and interactive signal/cue sync,
- `docs/CHANGELOG.md` already records the user-visible bounded change,
- shipped commit found in git history: `193d752 feat: improve tag filter visibility`.

### Slice 2 — Tag reset/default clarity guard

**Status:** implemented

**Intent:** Sprawdzić, czy reset z aktywnego taga do `All tags` pozostaje wystarczająco czytelny i deterministyczny także po interakcji użytkownika oraz programmatic refresh, bez dokładania nowej logiki filtrów.

**Why this next:**
- it stays on the exact same seam,
- it is the cheapest bounded follow-up after the shipped visibility label,
- it can land as either a tiny runtime fix or a test-only guard slice,
- it strengthens trust in the neutral/default state of the prompt library.

**Implemented:**
- added one focused regression guard for the active-tag -> neutral reset path in `tests/test_prompt_filter_panel.py`,
- confirmed the existing `PromptFilterPanel` runtime already restores `Tag filter: all tags` when the tag combo returns to index `0`,
- confirmed the same user interaction still emits exactly one `filters_changed` signal carrying the restored calm label text,
- kept the slice test-only because no runtime change was needed.

**Good shape:**
- active tag -> reset to `All tags` restores `Tag filter: all tags`,
- repopulating tags while the neutral state is selected keeps the helper calm and correct,
- no duplicate emissions or extra filter semantics churn,
- if the first RED probe already passes, close as guard-only / covered by existing behavior.

**Likely files:**
- Modify: `tests/test_prompt_filter_panel.py`
- Maybe modify: `gui/widgets/prompt_filter_panel.py`
- Maybe modify: `docs/CHANGELOG.md` only if user-visible wording/behavior changes

**Verification:**
- targeted probe: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py::test_tag_filter_panel_reset_restores_all_tags_visibility_cue -q` -> `1 passed`, so the runtime seam was already correct,
- targeted suite: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py -q` -> `4 passed`,
- nearby smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py tests/test_prompt_list_model.py -q` -> `20 passed`,
- lint: `.venv/bin/ruff check gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py` -> `All checks passed!`,
- syntax: `python -m py_compile gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py` -> `OK`.

### Slice 3 — Nearby filter-state continuity probe

**Status:** implemented

**Intent:** Zobaczyć, czy najbliższy istniejący filter-panel seam potrzebuje jeszcze jednego małego continuity cue wspierającego czytanie listy promptów, ale bez tworzenia drugiego systemu statusów.

**Implemented:**
- added one bounded continuity cue on the existing `PromptFilterPanel` seam so active search now surfaces `Sort locked during search` beside the sort control,
- kept the cue subordinate to the existing search behavior: the sort combo still disables during active search exactly as before,
- kept the slice local to the filter-panel + search-controller seam, without presenter/coordinator rewrite, persistence changes, or new retrieval status layers.

**Decision rule:**
- only proceed if repo recon finds one explicit existing UI seam and one real legibility gap,
- if the first candidate RED test passes immediately, close as `covered by existing behavior` and do not force runtime churn.

**Verification:**
- RED confirmed first with `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py::test_tag_filter_panel_active_search_disables_sort_with_visible_continuity_cue -q` -> `1 failed` because the sort lock had no visible continuity cue,
- GREEN after the minimal panel-only change with the same targeted pytest command -> `1 passed`,
- targeted suite: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py -q` -> `5 passed`,
- nearby smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py tests/test_prompt_list_model.py -q` -> `21 passed`,
- lint: `.venv/bin/ruff check gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py` -> `All checks passed!`,
- syntax: `python -m py_compile gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py` -> `OK`.

**Possible shape:**
- one compact helper text refinement near existing filter controls,
- no presenter/coordinator rewrite,
- no duplication of prompt-list retrieval status messages.

### Slice 4 — Filter cue locality / guard closure

**Status:** implemented

**Intent:** Domknąć filter-clarity cycle guardami reset/locality i jasno oznaczyć, czy filter cues pozostają GUI-local by design.

**Implemented:**
- extended the existing GUI-local parity guard suite in `tests/test_retrieval_cues_parity.py` so the filter-panel helper cues now have an explicit locality test beside the earlier prompt-list retrieval cue guard,
- confirmed `Tag filter: ...` and `Sort locked during search` remain widget-local labels on `PromptFilterPanel`,
- confirmed these filter cues do not surface as shared analytics/headless fields like `decision_summary`, `next_action_summary`, or `freshness_summary`,
- kept the slice test-only because the runtime seam already behaved correctly and no CLI/headless churn was justified.

**Verification:**
- RED probe first: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_retrieval_cues_parity.py -q` -> initial error (`fixture 'qtbot' not found`), then the focused guard was rewritten to use a local `QApplication` fixture and became a valid locality probe,
- targeted locality suite: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_retrieval_cues_parity.py -q` -> `2 passed`,
- nearby smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py tests/test_prompt_list_model.py tests/test_retrieval_cues_parity.py -q` -> `23 passed`,
- lint/format: `ruff check tests/test_retrieval_cues_parity.py && ruff format tests/test_retrieval_cues_parity.py --check` -> `All checks passed!`,
- syntax: `python -m py_compile tests/test_retrieval_cues_parity.py gui/widgets/prompt_filter_panel.py` -> `OK`.

**Good shape:**
- verify the cues stay inside the GUI filter-panel seam,
- add no-op or parity guards only where a real regression risk exists,
- avoid CLI/headless churn unless the cue enters shared analytics/product truth.

---

## Recommended first slice

### Pick first: Slice 2 — Tag reset/default clarity guard

Why this first:
- Slice 1 is already shipped and gives a clean baseline,
- the next cheapest confidence/trust gain is to lock the neutral reset path,
- it keeps the cycle bounded to one existing seam,
- it can naturally become either a tiny runtime fix or a test-only guard without fake churn.

---

## Implementation brief for the next slice

### Task 1: Reconfirm the shipped tag-visibility seam

**Objective:** Read the current `PromptFilterPanel` helper-label logic and the nearest focused tests before adding the next probe.

**Files:**
- Read: `gui/widgets/prompt_filter_panel.py`
- Read: `tests/test_prompt_filter_panel.py`
- Read: `docs/implementation-brief-2026-04-28-tag-filter-visibility-v1.md`

**Verification:**
- confirm the current helper label updates on `set_tags(...)` and `_handle_tag_changed()`,
- confirm the existing tests do not yet lock the active->reset path explicitly.

### Task 2: Write a RED test for active-tag reset clarity

**Objective:** Prove whether returning from an active tag to `All tags` still keeps the helper cue and signal posture correct.

**Files:**
- Modify: `tests/test_prompt_filter_panel.py`

**Verification:**
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py::test_tag_filter_panel_reset_restores_all_tags_visibility_cue -q`
- Expected before implementation: either one focused failure (real gap) or immediate pass (guard-only closure signal).

### Task 3: If RED fails, implement the smallest seam-local fix

**Objective:** Repair only the visible helper/reset behavior inside `PromptFilterPanel`.

**Files:**
- Maybe modify: `gui/widgets/prompt_filter_panel.py`
- Modify: `tests/test_prompt_filter_panel.py`

**Verification:**
- rerun the targeted test,
- then run `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py -q`,
- run `.venv/bin/ruff check gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py`,
- run `python -m py_compile gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py`.

### Task 4: If RED passes immediately, close as a guard-only slice

**Objective:** Avoid fake runtime churn when the reset path is already correct.

**Files:**
- Modify: this roadmap
- Maybe modify: `docs/CHANGELOG.md` only if the new regression coverage materially deserves an execution note

**Verification:**
- record the exact targeted pytest command/result,
- mark the slice as `implemented` (test-only guard) or `covered by existing behavior` based on the real outcome.

### Task 5: Sync the ledger before any commit/push

**Objective:** Keep this roadmap as the canonical execution ledger for the new filter-clarity cycle.

**Files:**
- Modify: this file
- Modify: `docs/CHANGELOG.md` only if user-visible behavior changed or the execution ledger needs a short trace
- Update `docs/product-ssot.md` only if product posture or priority order changed

---

## Documentation rule for this roadmap

Po każdej implementacji w tym cyklu:
1. najpierw zaktualizować ten roadmap,
2. dopisać krótki implemented / verified note,
3. zaktualizować `docs/CHANGELOG.md` tylko jeśli zmieniło się user-visible behavior albo execution ledger wymaga krótkiego śladu,
4. aktualizować `docs/product-ssot.md` tylko jeśli zmienia się definicja produktu albo porządek priorytetów.

SSOT ma pozostać stabilny; ten plik ma być żywym execution ledgerem.

---

## Definition of done for this cycle

Ten cykl będzie można uznać za dobrze domknięty, gdy PromptManager:
- wyraźniej komunikuje aktywne zawężenie biblioteki promptów,
- zachowuje spokojne i deterministyczne default/reset cues dla filtrów,
- wspiera czytanie listy promptów bez budowania drugiego systemu statusów,
- zostaje przy asset-first posture,
- i nadal jasno rozróżnia, które filter cues są shared truth, a które pozostają GUI-local by design.
