# PromptManager Capture / Reopen / Refine Entry Clarity Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć kolejny bounded execution cycle w Stage 2 tak, aby istniejące entry points do pracy z prompt assetami — `Quick Capture`, `Recent Prompts`, i `Promote Draft` — były spokojniejsze operacyjnie i lepiej wspierały przejście capture -> reopen -> refine bez zmiany storage, rankingu, promote semantics ani dodawania nowego workflow.

**Architecture:** Ten roadmap zostaje w asset-first core loop i celowo używa tylko istniejących seamów wokół `gui/dialogs/quick_capture.py`, `gui/dialogs/recent_prompts.py`, `gui/dialogs/draft_promote.py` oraz ich najbliższych testów. Zamiast przebudowy dialogów, wizardów albo nowych metadata panels, skupia się na bounded clarity cues przy wejściu do pracy: lepsza widoczność tego, co stanie się po capture/reopen/promote, czytelniejsze lightweight summaries i spokojne guardy tam, gdzie runtime już zachowuje się dobrze.

**Tech Stack:** Python 3.13, PySide6, pytest, Ruff, existing prompt dialog seams, PromptManager SSOT docs under `docs/`.

---

## Why this roadmap exists

Live repo/docs/code review confirms:
- `docs/product-roadmap-ssot.md` nadal utrzymuje **Stage 2 — Core prompt asset loop quality** jako priorytet,
- ostatnie bounded cycles dla prompt-library filter clarity, favorites clarity, i prompt-actions clarity są już praktycznie domknięte,
- repo jest clean (`master...origin/master` bez lokalnych zmian) i gotowe na nowy execution ledger,
- PromptManager ma trzy istniejące, bliskie centrum produktu entry seams, które są już użyteczne, ale były poprawiane tylko fragmentarycznie: `Quick Capture`, `Recent Prompts`, i `Promote Draft`,
- changelog pokazuje wcześniejsze dostarczenie capability slices na tych seamach (draft creation, recent reopen, recent-row metadata visibility, draft similarity advisory, bounded normalize/cleanup passes), ale nie ma jeszcze świeżego roadmap cyklu skupionego stricte na **entry clarity** dla capture/reopen/refine start points.

So the next useful move is not kolejny filter/action micro-cycle i nie nowa retrieval branch, ale świeży bounded execution ledger skupiony na **capture / reopen / refine entry clarity**.

Short version:

**asset-first core -> library/filter/action polish -> entry clarity for capture/reopen/refine**

---

## Confirmed baseline before this cycle

Assume already delivered and do not reopen without a failing regression:
- `Recent Prompts` already has deterministic ordering and visible second-line metadata (`Modified ... • Category: ...`),
- `Quick Capture` already supports bounded body cleanup (fence unwrap, blockquote unwrap, prompt-label strip, single-turn `User:` strip) and shared title heuristics,
- `Draft Promote` already exposes bounded similar-match advisory cues (`Likely duplicate`, `Very close match`, preview fallback, selected-match reason cues),
- prompt-actions wording, filter-panel cues, and favorites clarity from the latest cycles remain the active baseline,
- no change in storage model, prompt ranking, similarity scoring semantics, or promote-as-new persistence path should be assumed.

This cycle should build on those seams instead of replacing them.

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager ma już działające wejścia do szybkiego złapania promptu, ponownego otwarcia niedawno używanego assetu i wypromowania draftu do reusable prompta, to jak zrobić te wejścia bardziej czytelnymi operacyjnie, żeby operator szybciej rozumiał co robi dany entry point i bezpieczniej przechodził do dalszej pracy, nadal bez budowania nowego workflow?

Nowy cykl wzmacnia trzy rzeczy:

1. **entry-point clarity** — operator ma szybciej widzieć, po co użyć `Quick Capture`, `Recent`, albo `Promote Draft`,
2. **capture-to-reopen continuity** — lightweight metadata i helper wording mają wspierać świadome wejście do dalszego inspect/refine,
3. **bounded guard closure** — jeśli runtime już zachowuje się dobrze, dokładamy test/ledger clarity zamiast fake runtime churn.

---

## Constraints

Do not add:
- new dialogs or wizard steps,
- draft management dashboard,
- richer metadata editor in entry dialogs,
- ranking / similarity algorithm changes,
- persistence/schema changes,
- new CLI/headless parity unless a cue becomes shared product truth,
- another status layer competing with prompt-list or detail surfaces.

Reuse first:
- `gui/dialogs/quick_capture.py`
- `gui/dialogs/recent_prompts.py`
- `gui/dialogs/draft_promote.py`
- focused tests under `tests/` for those dialogs and nearest action handlers

---

## Roadmap stages for this cycle

### Stage A — Capture entry clarity

Cel: sprawić, żeby `Quick Capture` lepiej komunikował, jaki typ inputu jest oczekiwany i jaki bounded cleanup already happens, bez dokładania nowego preprocessing workflow.

Obszary:
- one compact operator-facing cue around accepted raw input posture,
- maybe one placeholder or helper-copy refinement,
- no change in normalization semantics.

Constraint:
- keep the dialog minimal,
- wording or tiny local helper only.

### Stage B — Reopen entry clarity

Cel: sprawić, żeby `Recent Prompts` lepiej wspierało świadomy reopen wyborem bez rozbudowy w history browser.

Obszary:
- maybe one compact visible cue beyond the existing metadata second line,
- selection/open wording sanity,
- no sorting or persistence changes.

Constraint:
- keep the dialog compact,
- use the existing row widget seam first.

### Stage C — Draft-to-refine entry clarity

Cel: upewnić się, że `Promote Draft` lepiej komunikuje operatorowi znaczenie advisory entry pointu bez przechodzenia w compare screen albo duplicate-management workflow.

Obszary:
- one compact advisory/entry wording refinement or nearby guard,
- keep `Promote as New` vs `Open Existing Match` semantics unchanged,
- no similarity engine changes.

Constraint:
- stay on the existing summary/list/button seam,
- prefer wording over structure.

### Stage D — Guard/locality closure

Cel: domknąć cycle małymi guardami i jasno zaznaczyć, które entry cues pozostają dialog-local by design.

Obszary:
- no-op closures where runtime already behaves correctly,
- explicit locality boundary if a cue stays dialog-only,
- avoid widening into shared analytics or SSOT churn.

Constraint:
- prefer test-only closure when runtime already does the right thing.

---

## Recommended execution order

1. inspect `Quick Capture` first for the cheapest real clarity gap on the entry seam,
2. then probe one compact `Recent Prompts` reopen cue only if it stays calmer than a history browser,
3. then inspect one bounded `Promote Draft` entry wording/guard seam,
4. close with guard/locality coverage.

---

## Candidate bounded slices

### Slice 1 — Quick Capture raw-input posture cue v1

**Status:** implemented

**Intent:** Dodać w istniejącym `Quick Capture` dialog jeden mały, operator-facing cue wyjaśniający, że to entry point dla surowego promptu / query i że PromptManager robi tylko bounded cleanup obvious wrappers, bez tworzenia nowego preprocessingu albo transcript parsera.

**Why this first:**
- jest to najbliższy entry seam w centrum asset-first workflow,
- istniejący dialog ma już cleanup semantics, ale komunikuje je głównie przez placeholder `Paste the raw prompt or LLM query here…`,
- to najtańszy kandydat na mały user-visible clarity gain bez zmiany logiki,
- da się łatwo zweryfikować na jednym dialog/test seamie.

**Implemented:**
- added one compact always-visible `_entry_guidance_label` on the existing `QuickCaptureDialog` seam,
- the cue now says `Paste a raw prompt or query. PromptManager only cleans obvious outer wrappers before saving the draft.`,
- kept all quick-capture cleanup semantics unchanged (fence unwrap, blockquote unwrap, prompt-label strip, single-turn `User:` strip),
- kept the slice local to the dialog wording seam without changing storage, validation flow, or adding new controls.

**Good shape:**
- one compact helper copy near the body input or dialog summary,
- wording stays calm and deterministic,
- no extra buttons, toggles, or advanced parsing options,
- existing cleanup behavior remains exactly the same.

**Likely files:**
- Maybe modify: `gui/dialogs/quick_capture.py`
- Modify: `tests/test_quick_capture_dialog.py`
- Maybe add brief: `docs/implementation-brief-2026-04-29-quick-capture-entry-clarity-v1.md`
- Modify: `docs/CHANGELOG.md` only if user-visible wording changes

**Verification target:**
- targeted offscreen pytest for one new `QuickCaptureDialog` wording test,
- nearby smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_quick_capture_dialog.py -q`,
- lint: `.venv/bin/ruff check gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py`,
- syntax: `python -m py_compile gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py`.

**Verified:**
- RED: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_quick_capture_dialog.py::test_quick_capture_dialog_shows_raw_input_entry_guidance -q` -> `1 failed` (`AttributeError: 'QuickCaptureDialog' object has no attribute '_entry_guidance_label'`),
- GREEN: same targeted test -> `1 passed`,
- targeted suite: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_quick_capture_dialog.py -q` -> `29 passed`,
- quality: `.venv/bin/ruff check gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py && python -m py_compile gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py` -> `All checks passed!`.

### Slice 2 — Recent reopen intent cue probe

**Status:** implemented

**Intent:** Sprawdzić, czy `Recent Prompts` potrzebuje jeszcze jednego małego cue pomagającego operatorowi szybciej zrozumieć reopen posture, ponad już dostarczone `Modified ... • Category: ...`, ale nadal bez rozbudowy w history browser.

**Implemented:**
- tightened the existing dialog summary wording so `Recent Prompts` now says `Reopen one of the prompts you touched most recently to continue refining it.`,
- kept the change local to the summary label seam and left row metadata, ordering, selection flow, and persistence unchanged,
- kept the slice bounded to reopen posture clarity without introducing a history browser or extra list controls.

**Good shape:**
- one compact visible line or wording refinement on the existing row/summary seam,
- no extra columns, filters, or sort modes,
- no runtime churn if the current reopen dialog is already calm enough.

**Likely files:**
- Maybe modify: `gui/dialogs/recent_prompts.py`
- Modify: `tests/test_recent_prompts.py`
- Maybe modify: `docs/CHANGELOG.md`

**Verified:**
- RED: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_recent_prompts.py::test_recent_prompts_dialog_summary_mentions_reopen_for_further_work -q` -> `1 failed` because the summary still ended at `most recently.`,
- GREEN: same targeted test -> `1 passed`,
- targeted suite: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_recent_prompts.py -q` -> `4 passed`,
- quality: `.venv/bin/ruff check gui/dialogs/recent_prompts.py tests/test_recent_prompts.py && python -m py_compile gui/dialogs/recent_prompts.py tests/test_recent_prompts.py` -> `All checks passed!`.

### Slice 3 — Promote Draft advisory entry wording probe

**Status:** proposed

**Intent:** Sprawdzić, czy `Promote Draft` potrzebuje jednego małego entry-level wording refinementu lub guardu na summary/button seamie, żeby advisory path był czytelniejszy bez ruszania similarity semantics.

**Good shape:**
- one compact summary/button copy refinement or a guard that locks the existing good behavior,
- no new compare flow,
- no duplicate-management branching,
- no change in `Promote as New` / `Open Existing Match` action semantics.

**Likely files:**
- Maybe modify: `gui/dialogs/draft_promote.py`
- Modify: `tests/test_draft_promote_dialog.py`
- Maybe modify: `docs/CHANGELOG.md`

### Slice 4 — Entry-cue locality / guard closure

**Status:** proposed

**Intent:** Domknąć cycle lekkimi guardami i jasno oznaczyć, które cues pozostają local to dialog seams zamiast becoming shared product truth.

**Good shape:**
- keep guard-only if runtime already behaves correctly,
- no CLI/headless widening,
- update the roadmap ledger with explicit locality decisions.

---

## Recommended first slice

### Pick first: Slice 1 — Quick Capture raw-input posture cue v1

Why this first:
- nearest asset-entry seam,
- smallest likely wording-only slice with visible operator payoff,
- no evidence yet that it is already locked by a focused regression,
- cheap to verify and easy to keep bounded.

---

## Implementation brief for the next slice

### Task 1: Reconfirm the current quick-capture seam

**Objective:** Read the exact `QuickCaptureDialog` body-input/summary wording and nearest tests before adding the first probe.

**Files:**
- Read: `gui/dialogs/quick_capture.py`
- Read: `tests/test_quick_capture_dialog.py`
- Read: `docs/CHANGELOG.md` quick-capture-related entries

**Verification:**
- confirm the current dialog exposes only placeholder-level input guidance,
- confirm the existing tests lock cleanup semantics but not yet a compact entry-clarity cue.

### Task 2: Write a RED test for the quick-capture entry cue

**Objective:** Prove whether `Quick Capture` still needs one explicit operator-facing raw-input posture cue.

**Files:**
- Modify: `tests/test_quick_capture_dialog.py`

**Verification:**
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_quick_capture_dialog.py::<new_test_name> -q`
- Expected before implementation: one focused failure (real wording gap) or immediate pass (guard-only closure signal).

### Task 3: If RED fails, implement the smallest seam-local fix

**Objective:** Repair only the bounded entry wording inside `QuickCaptureDialog`.

**Files:**
- Maybe modify: `gui/dialogs/quick_capture.py`
- Modify: `tests/test_quick_capture_dialog.py`
- Maybe add: `docs/implementation-brief-2026-04-29-quick-capture-entry-clarity-v1.md`

**Verification:**
- rerun the targeted test,
- then run `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_quick_capture_dialog.py -q`,
- run `.venv/bin/ruff check gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py`,
- run `python -m py_compile gui/dialogs/quick_capture.py tests/test_quick_capture_dialog.py`.

### Task 4: If RED passes immediately, close as a guard-only slice

**Objective:** Avoid fake runtime churn when the quick-capture entry seam is already sufficiently clear.

**Files:**
- Modify: this roadmap
- Maybe modify: `docs/CHANGELOG.md` only if the new regression coverage materially deserves an execution note

**Verification:**
- record the exact targeted pytest command/result,
- mark the slice as `implemented` (test-only guard) or `covered by existing behavior` based on the real outcome.

### Task 5: Sync the ledger before any commit/push

**Objective:** Keep this roadmap as the canonical execution ledger for the new entry-clarity cycle.

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
- czytelniej komunikuje entry posture dla capture / reopen / draft-promote seams,
- zachowuje spokojne i deterministyczne wording/local cues przy wejściu do dalszej pracy,
- wspiera capture -> reopen -> refine bez budowania nowego workflow layer,
- zostaje przy asset-first posture,
- i nadal jasno rozróżnia, które entry cues są shared truth, a które pozostają dialog-local by design.
