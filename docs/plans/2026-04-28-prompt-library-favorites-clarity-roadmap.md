# PromptManager Prompt-Library Favorites Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć kolejny bounded execution cycle po filter-clarity tak, aby istniejący mechanizm favorites w bibliotece promptów był czytelniejszy operacyjnie i lepiej wspierał retrieve -> inspect -> reuse bez zmiany semantyki filtrowania, rankingu ani persistence.

**Architecture:** Ten roadmap zostaje w asset-first core loop i celowo używa tylko istniejących seamów wokół `PromptFilterPanel`, `PromptDetailWidget`, `main_window_handlers`, oraz najbliższych prompt-library consumers. Zamiast przebudowy favorites w collections/pinning workflow albo dodawania dashboardów, skupia się na bounded visibility cues, active-state continuity i lekkich guardach reset/locality tam, gdzie operatorowi brakuje jawnej informacji, że pracuje na ulubionych promptach.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, existing prompt-library filter/detail flows, PromptManager SSOT docs under `docs/`.

---

## Why this roadmap exists

Live repo/docs/code review confirms:
- `docs/product-ssot.md` nadal utrzymuje asset-first posture i wprost dopuszcza `light collections or favorites` tylko wtedy, gdy realnie poprawiają retrieval speed i redukują clutter,
- cykl `docs/plans/2026-04-28-prompt-library-filter-clarity-roadmap.md` został właśnie domknięty: slice 1–4 są `implemented`, a repo jest clean po commit/push,
- PromptManager ma już istniejący favorites seam blisko głównego celu produktu: `Favorites only` w `gui/widgets/prompt_filter_panel.py`, favorite toggle w `gui/widgets/prompt_detail_widget.py`, oraz list-level filtering w `gui/prompt_list_coordinator.py`,
- changelog pokazuje wcześniejsze `Favorites v1`, ale ten zakres był głównie storage/filter capability + regression coverage, nie osobny readability/usability cycle,
- w aktualnym filter panelu favorites nadal są mniej jawne operacyjnie niż świeżo dopracowane tag/sort cues: checkbox działa, ale nie ma jeszcze porównywalnie czytelnego active-state visibility path.

So the next useful move is not kolejny retrieval/discovery extension i nie kolejny inspect/detail wording pass, ale świeży bounded execution ledger skupiony na **prompt-library favorites clarity**.

Short version:

**asset-first core -> retrieval confidence -> filter clarity -> favorites clarity**

---

## Confirmed baseline before this cycle

Assume already delivered and do not reopen without a failing regression:
- prompt-local favorite persistence and detail toggle behavior,
- `Favorites only` filtering in the existing prompt-library filter panel,
- bounded retrieval/discovery confidence cues in prompt-list seams,
- recent prompt reopen metadata rows,
- bounded context-menu clarity for prompt actions,
- prompt-library filter clarity cues for `Tag filter: ...` and `Sort locked during search`,
- no change in favorites semantics, ranking, search scoring, or storage model.

This cycle should build on the existing favorites seam instead of replacing it.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager ma już działające favorites jako lekki retrieval accelerator, to jak zrobić ten stan bardziej czytelnym i spokojniejszym operacyjnie, żeby operator szybciej widział, kiedy pracuje na ulubionych promptach, bez rozbudowy w collections, pinning albo drugi system statusów?

Nowy cykl wzmacnia trzy rzeczy:

1. **favorites visibility** — operator ma szybciej widzieć, że biblioteka jest zawężona do favorites,
2. **detail-to-list continuity** — favorite toggle w detail view ma lepiej łączyć się mentalnie z widokiem listy i filtrem,
3. **bounded guard closure** — reset/default/locality mają zostać jawnie sprawdzone bez fake runtime churn.

---

## Constraints

Do not add:
- collections,
- pinned prompts / starred sets,
- favorites ranking boosts,
- new persistence for favorites UI state,
- dashboard or analytics for favorites,
- CLI/headless parity unless a cue becomes shared product truth,
- another recommendation/status layer competing with the prompt list.

Reuse first:
- `gui/widgets/prompt_filter_panel.py`
- `gui/widgets/prompt_detail_widget.py`
- `gui/main_window_handlers.py`
- `gui/prompt_list_coordinator.py`
- focused tests under `tests/` for filter panel, detail widget, and main-window favorites paths

---

## Roadmap stages for this cycle

### Stage A — Visible favorites-filter posture

Cel: sprawić, żeby aktywny stan `Favorites only` był czytelny bez zmiany logiki filtrowania.

Obszary:
- explicit favorites-only visibility cue,
- calm default state for the non-favorites view,
- no semantic change to checkbox/filter behavior.

Constraint:
- stay inside the filter-panel seam first,
- no ranking or filtering rewrite.

### Stage B — Detail-to-list continuity

Cel: upewnić się, że operator po favorite toggle lepiej rozumie wpływ na bibliotekę promptów i retrieval.

Obszary:
- one compact continuity cue or status refinement,
- no workflow branching,
- no extra dialog/banner system.

Constraint:
- keep the detail toggle simple,
- prefer wording/local cue over behavior change.

### Stage C — Reset/default-state confidence

Cel: domknąć neutralny/default favorites state lekkimi guardami i upewnić się, że active -> off reset pozostaje spokojny i deterministyczny.

Obszary:
- reset to non-favorites-only wording/cue behavior,
- guard coverage for programmatic favorite filter changes,
- no extra reset controls.

Constraint:
- prefer tests/guards if runtime already behaves correctly.

### Stage D — Guard/locality closure

Cel: domknąć cycle parity/reset/no-op checks i jasno oznaczyć, które favorites cues pozostają GUI-local by design.

Obszary:
- locality decision for favorites helper cues,
- no-op closure if candidate slice is already covered,
- explicit boundary against shared analytics/headless fields.

Constraint:
- prefer test-only closure when runtime already does the right thing.

---

## Recommended execution order

1. inspect the existing `Favorites only` seam and lock the current baseline,
2. add one small active-state visibility improvement if a real bounded gap exists,
3. probe one continuity slice between detail toggle and list/filter state only if it stays calmer than a second status layer,
4. close reset/locality coverage.

---

## Candidate bounded slices

### Slice 1 — Favorites-only active-state visibility v1

**Status:** implemented

**Intent:** Dodać w istniejącym `PromptFilterPanel` mały, zawsze widoczny cue pokazujący, czy biblioteka jest zawężona do ulubionych promptów, bez zmiany logiki favorites filtering.

**Why this first:**
- it stays on the same seam just refined for tags/sort continuity,
- it is the closest unresolved usability gap in the current favorites flow,
- it improves retrieval clarity without widening into new feature semantics,
- it can land as a tiny runtime cue or a test-only closure if the state is already sufficiently visible.

**Good shape:**
- calm default cue when favorites filter is off,
- explicit active cue when `Favorites only` is on,
- no extra persistence,
- no list ranking/order changes,
- no conflict with existing tag/sort helper cues.

**Likely files:**
- Modify: `gui/widgets/prompt_filter_panel.py`
- Modify: `tests/test_prompt_filter_panel.py`
- Maybe modify: `docs/CHANGELOG.md`

**Verification target:**
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py::test_favorites_filter_panel_shows_active_visibility_cue -q`
- nearby smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py tests/test_prompt_list_model.py -q`
- lint: `.venv/bin/ruff check gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py`
- syntax: `python -m py_compile gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py`

**Implemented notes:**
- added one always-visible `favoritesFilterVisibilityLabel` on the existing `PromptFilterPanel` seam,
- default cue now reads `Favorites filter: all prompts`,
- active cue now reads `Favorites filter: favorites only`,
- kept `Favorites only` filtering semantics unchanged and routed checkbox interaction through a local label-update helper before emitting `filters_changed`.

**Verified:**
- RED: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py::test_favorites_filter_panel_shows_active_visibility_cue -q` -> failed as expected before implementation,
- GREEN: same targeted test -> `1 passed`,
- smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py tests/test_prompt_list_model.py -q` -> `22 passed`,
- lint: `.venv/bin/ruff check gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py` -> `All checks passed!`,
- syntax: `python -m py_compile gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py` -> OK.

### Slice 2 — Favorite toggle continuity hint probe

**Status:** implemented

**Intent:** Sprawdzić, czy favorite toggle w detail view potrzebuje jednego małego continuity hintu pomagającego operatorowi zrozumieć wpływ na późniejszy retrieval/list filtering, ale bez rozbudowy w nowy workflow.

**Good shape:**
- one compact wording/status refinement on the existing detail/main-window seam,
- no extra confirmation dialog,
- no change to favorite persistence semantics,
- no duplicate cue if the filter panel already provides enough clarity.

**Likely files:**
- Maybe modify: `gui/widgets/prompt_detail_widget.py`
- Maybe modify: `gui/main_window_handlers.py`
- Modify: nearest favorites-related tests under `tests/`
- Maybe modify: `docs/CHANGELOG.md`

**Implemented notes:**
- refined the existing detail-view favorite tooltip to make the retrieval/list impact explicit,
- add-state wording now points to later discovery through `Favorites only`,
- kept favorite persistence, button behavior, and status semantics unchanged.

**Verified:**
- RED: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_detail_widget.py::test_prompt_detail_widget_toggles_favorite_action_from_detail_flow -q` -> failed as expected before implementation,
- GREEN: same targeted test -> `1 passed`,
- smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_detail_widget.py tests/test_prompt_filter_panel.py -q` -> `39 passed`,
- lint: `.venv/bin/ruff check gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> `All checks passed!`,
- syntax: `python -m py_compile gui/widgets/prompt_detail_widget.py tests/test_prompt_detail_widget.py` -> OK.

### Slice 3 — Favorites reset/default clarity guard

**Status:** implemented

**Intent:** Sprawdzić, czy przejście z `Favorites only` back to neutral library state pozostaje wystarczająco czytelne i deterministyczne po interakcji użytkownika oraz programmatic changes, bez dokładania nowej logiki filtrów.

**Good shape:**
- active favorites -> reset restores calm default cue,
- no duplicate emissions or extra filter semantics churn,
- if the first RED probe already passes, close as a guard-only slice.

**Likely files:**
- Modify: `tests/test_prompt_filter_panel.py`
- Maybe modify: `gui/widgets/prompt_filter_panel.py`
- Maybe modify: `docs/CHANGELOG.md` only if user-visible behavior changes

**Implemented notes:**
- added one focused guard covering `favorites only -> off` reset on the existing filter-panel seam,
- confirmed the calm default cue returns as `Favorites filter: all prompts`,
- confirmed the interactive reset still emits exactly one `filters_changed` event carrying the restored default cue,
- runtime remained unchanged because the seam already behaved correctly.

**Verified:**
- probe: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py::test_favorites_filter_panel_reset_restores_default_visibility_cue -q` -> `1 passed` immediately,
- smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py tests/test_prompt_detail_widget.py -q` -> `40 passed`,
- lint: `.venv/bin/ruff check tests/test_prompt_filter_panel.py` -> `All checks passed!`,
- syntax: `python -m py_compile tests/test_prompt_filter_panel.py` -> OK.

### Slice 4 — Favorites cue locality / guard closure

**Status:** implemented

**Intent:** Domknąć favorites-clarity cycle guardami reset/locality i jasno oznaczyć, czy favorites cues pozostają GUI-local by design.

**Good shape:**
- verify the cues stay inside the GUI filter/detail seam,
- add no-op or parity guards only where a real regression risk exists,
- avoid CLI/headless churn unless the cue enters shared analytics/product truth.

**Implemented notes:**
- extended the existing GUI-local parity guard to cover the favorites helper cue on the same `PromptFilterPanel` seam,
- confirmed `Favorites filter: favorites only` remains widget-local alongside `Tag filter: ...` and `Sort locked during search`,
- confirmed no shared/headless analytics fields (`decision_summary`, `next_action_summary`, `freshness_summary`) were introduced for these favorites cues,
- runtime remained unchanged because the favorites cue was already local by design.

**Verified:**
- probe: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_retrieval_cues_parity.py::test_prompt_filter_panel_cues_remain_local_widget_state -q` -> `1 passed` immediately,
- smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_retrieval_cues_parity.py tests/test_prompt_filter_panel.py tests/test_prompt_list_model.py -q` -> `25 passed`,
- lint: `.venv/bin/ruff check tests/test_retrieval_cues_parity.py` -> `All checks passed!`,
- syntax: `python -m py_compile tests/test_retrieval_cues_parity.py` -> OK.

---

## Recommended first slice

### Pick first: Slice 1 — Favorites-only active-state visibility v1

Why this first:
- it is the nearest unresolved readability gap on an already-existing core seam,
- it matches the just-finished filter-clarity cycle without reopening it artificially,
- it is cheap to verify with targeted filter-panel tests,
- it preserves the asset-first posture while making favorites feel more intentional as a retrieval aid.

---

## Implementation brief for the next slice

### Task 1: Reconfirm the current favorites filter seam

**Objective:** Read the current `PromptFilterPanel` favorites checkbox behavior and the nearest focused tests before adding the first probe.

**Files:**
- Read: `gui/widgets/prompt_filter_panel.py`
- Read: `tests/test_prompt_filter_panel.py`
- Read: `docs/CHANGELOG.md` favorites-related entries

**Verification:**
- confirm `Favorites only` currently changes filtering semantics without an explicit helper cue comparable to tag/sort visibility labels,
- confirm the nearest tests do not yet lock a favorites active-state helper cue.

### Task 2: Write a RED test for favorites active-state visibility

**Objective:** Prove whether enabling `Favorites only` still needs one explicit visibility cue on the filter-panel seam.

**Files:**
- Modify: `tests/test_prompt_filter_panel.py`

**Verification:**
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py::test_favorites_filter_panel_shows_active_visibility_cue -q`
- Expected before implementation: either one focused failure (real gap) or immediate pass (guard-only closure signal).

### Task 3: If RED fails, implement the smallest seam-local fix

**Objective:** Repair only the visible favorites helper behavior inside `PromptFilterPanel`.

**Files:**
- Maybe modify: `gui/widgets/prompt_filter_panel.py`
- Modify: `tests/test_prompt_filter_panel.py`

**Verification:**
- rerun the targeted test,
- then run `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_filter_panel.py -q`,
- run `.venv/bin/ruff check gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py`,
- run `python -m py_compile gui/widgets/prompt_filter_panel.py tests/test_prompt_filter_panel.py`.

### Task 4: If RED passes immediately, close as a guard-only slice

**Objective:** Avoid fake runtime churn when the favorites visibility path is already correct.

**Files:**
- Modify: this roadmap
- Maybe modify: `docs/CHANGELOG.md` only if the new regression coverage materially deserves an execution note

**Verification:**
- record the exact targeted pytest command/result,
- mark the slice as `implemented` (test-only guard) or `covered by existing behavior` based on the real outcome.

### Task 5: Sync the ledger before any commit/push

**Objective:** Keep this roadmap as the canonical execution ledger for the new favorites-clarity cycle.

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
- wyraźniej komunikuje, kiedy biblioteka jest zawężona do favorites,
- zachowuje spokojne i deterministyczne default/reset cues dla favorites state,
- wspiera retrieval -> inspect -> reuse bez budowania drugiego systemu statusów,
- zostaje przy asset-first posture,
- i nadal jasno rozróżnia, które favorites cues są shared truth, a które pozostają GUI-local by design.
