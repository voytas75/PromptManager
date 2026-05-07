# PromptManager Prompt Chain Rollout Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Domknąć prompt chains jako bounded, czytelny i przewidywalny linear runner: bez workflow-engine drift, z lepszą walidacją, spójną semantyką runów, wygodniejszą konsumpcją wyników i lekką historią uruchomień.

**Architecture:** Utrzymaj obecny model liniowego uruchamiania i nie poszerzaj semantyki silnika. Najpierw oczyść kontrakt i narrację modelu, potem dołóż małe operator-facing seams w CLI/GUI, a na końcu lekką trwałą historię opartą o bounded evidence. Każdy slice ma być mały, testowalny i zgodny z SSOT `docs/plans/2026-05-06-prompt-chain-ssot.md`.

**Tech Stack:** Python 3.13+, PromptManager core/CLI/GUI, pytest, Ruff, Pyright strict, istniejące docs/SSOT pod `docs/plans/`.

**Companion bounded execution plan:** `docs/plans/2026-05-06-prompt-chain-semantic-slices-plan.md`

---

## Scope guardrails

Ten plan może poprawiać tylko:
- model clarity i jawne oznaczenie legacy fields,
- validate/apply dry-run ergonomics,
- bounded result-consumption modes,
- editor ergonomics bez workflow-canvas,
- lightweight persisted history,
- minimal docs sync między SSOT a aktywnym ledgerem.
- run-semantics alignment między backend, CLI, GUI i history seams.

Ten plan nie może dodawać:
- branchingu,
- loops,
- retry workflow semantics,
- scheduler-first chains,
- visual workflow buildera,
- chain analytics dashboardu,
- multi-agent orchestration,
- osobnego variables engine dla chains.

---

## Confirmed baseline

Traktuj jako już dostarczone:
- linear plain-text chain input,
- per-step piping previous output -> next input,
- GUI CRUD/run/inspect,
- CLI `list/show/export/validate/apply/run`,
- `prompt-chain-show --json`,
- `prompt-chain-run --json`,
- `--final-output-only`, `--summary-only`, `--status-only`,
- `--output-file`,
- web-search enrichment,
- final summary,
- explicit `final_output_text` vs `final_summary_text`,
- canonical `step_output_key`, explicit `step_aliases`, explicit `final_step_*`,
- GUI copy/save result actions,
- GUI session-only recent run history.

Potwierdzona luka po ostatnim review:
- `final_step_*` nie są jeszcze semantycznie domknięte względem terminalnego kroku wykonania,
- aggregate run status nie jest jeszcze jednym backend-owned contract konsumowanym wszędzie.

Nie planuj tych rzeczy ponownie jako nowych feature’ów.

---

## Phase 0 — Run semantics alignment

### Task 0: Sync semantic slices plan into active docs

**Status:** done

**Objective:** Ujawnić w SSOT i rollout ledgerze, że kolejnym aktywnym bounded workstreamem jest semantic cleanup dla run status i final-vs-terminal step.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`
- Reference: `docs/plans/2026-05-06-prompt-chain-semantic-slices-plan.md`

### Task 0A: Define run-status and final/terminal-step semantics

**Status:** planned

**Objective:** Ustalić jeden backend-owned contract dla `run_status`, `final_step_*` i `terminal_step_*`.

**Execution note:** Szczegółowy plan wykonawczy jest w `docs/plans/2026-05-06-prompt-chain-semantic-slices-plan.md` (Slice A).

### Task 0B: Align CLI and history consumers with backend run semantics

**Status:** planned

**Objective:** Usunąć lokalne heurystyki statusu/finalności z CLI i history surfaces.

**Execution note:** Szczegółowy plan wykonawczy jest w `docs/plans/2026-05-06-prompt-chain-semantic-slices-plan.md` (Slice A + Slice B).

---

## Phase 1 — Model clarity and legacy boundary

### Task 1: Sync docs so active ledger is unambiguous

**Status:** done

**Objective:** Usunąć drift dokumentacyjny między SSOT a równoległymi ledgerami, żeby kolejne slice’y miały jeden jasny execution source.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-implementation-plan.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-functional-improvement-implementation-plan.md`
- Optional: `docs/plans/2026-05-06-prompt-chain-next-slices-plan.md`

**Steps:**
1. Sprawdź, który plik ma zostać aktywnym execution ledgerem.
2. Oznacz pozostałe jako narrow slice note / superseded / supporting note.
3. Usuń sprzeczne wskazania „active implementation ledger”.
4. Zachowaj mały diff — bez przepisywania całego SSOT.

**Verification:**
- `search_files("active implementation ledger|Status: superseded|supporting note", path="docs/plans", target="content")`
- Oczekiwane: jeden aktywny ledger, brak sprzecznych pointerów.

**Implemented:**
- oznaczono `docs/plans/2026-05-06-prompt-chain-implementation-plan.md` jako supporting note zamiast active,
- dopisano jawny pointer do aktywnego execution ledger: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`,
- doprecyzowano, że `docs/plans/2026-05-06-prompt-chain-next-slices-plan.md` jest tylko forward-looking supporting note,
- zachowano `docs/plans/2026-05-06-prompt-chain-functional-improvement-implementation-plan.md` jako supporting note dla już domkniętych slices.

**Verified:**
- `search_files("active implementation ledger|Execution ledger|supporting note|rollout plan", path="docs/plans", target="content")`
- potwierdzono jeden aktywny ledger: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`
- usunięto sprzeczny status `active` z `docs/plans/2026-05-06-prompt-chain-implementation-plan.md`

---

### Task 2: Add failing model tests for explicit legacy/inactive semantics labeling

**Status:** completed

**Objective:** Zamrozić oczekiwane zachowanie zanim zmienisz model i inspect surfaces.

**Files:**
- Modify: `tests/test_prompt_chain_model.py`
- Modify: `tests/test_prompt_chain_cli.py`
- Reference: `models/prompt_chain_model.py`
- Reference: `cli/commands.py`

**Test targets:**
- legacy fields są zachowane dla compatibility/import boundaries,
- active defaults nie sugerują aktywnego workflow behavior,
- CLI/inspect potrafi ujawnić, że legacy fields są inactive semantics,
- export/show nie sprzedaje `input_template` / `condition` / `variables_schema` jako aktywnego runtime contract.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py -q`
Expected: FAIL na nowych asercjach.

---

### Task 3: Implement compatibility-safe legacy field labeling in model and CLI inspect surfaces

**Status:** completed

**Objective:** Oczyścić semantykę bez łamania import/export compatibility.

**Files:**
- Modify: `models/prompt_chain_model.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_model.py`
- Modify: `tests/test_prompt_chain_cli.py`
- Optional docs sync: `docs/README-DEV.md`

**Implementation targets:**
- zawęzić docstringi/nazewnictwo do bounded linear runner framing,
- zachować legacy fields na granicy storage/import,
- dodać jawny operator-facing sygnał, że to inactive/legacy semantics,
- nie zmieniać aktualnego runtime execution flow.

**Constraint:**
Nie kasuj od razu pól ze storage modelu, jeśli to może złamać round-trip.
Najpierw label + contract clarity.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py -q`
- `ruff check models/prompt_chain_model.py cli/commands.py tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py`
- `.venv/bin/pyright models/prompt_chain_model.py cli/commands.py tests/test_prompt_chain_model.py tests/test_prompt_chain_cli.py`

---

## Phase 2 — Validation and dry-run trust surfaces

### Task 4: Add failing tests for `prompt-chain-validate --json`

**Status:** completed

**Objective:** Zanim dodasz nowy output mode, zamroź jego minimalny report contract.

**Files:**
- Modify: `tests/test_prompt_chain_cli.py`
- Modify: `cli/parser.py`
- Reference: `cli/commands.py`

**Test targets:**
- `prompt-chain-validate --json` zwraca deterministic JSON,
- report zawiera co najmniej:
  - `valid`,
  - `warnings`,
  - `step_count`,
  - `legacy_fields_detected` lub równoważne bounded pole,
- valid i invalid payload mają przewidywalny shape.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
Expected: FAIL na nowym trybie.

---

### Task 5: Implement `prompt-chain-validate --json`

**Status:** functionally-complete / pyright-debt-followup

**Objective:** Dodać machine-readable validation surface bez tworzenia drugiego systemu semantycznego.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Implementation targets:**
- parser flag `--json` dla validate,
- deterministic JSON report dla validate,
- bounded warning list dla legacy/inactive semantics,
- zachować obecny human-readable path.

**Constraint:**
Jeden prosty report shape. Bez osobnych verbose schemas.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
- `ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`
- `.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`

---

### Task 6: Add failing tests for `prompt-chain-apply --dry-run`

**Status:** completed (already present in repo)

**Objective:** Zamrozić non-persistent preview behavior przed implementacją.

**Files:**
- Modify: `tests/test_prompt_chain_cli.py`
- Modify: `cli/parser.py`
- Reference: `cli/commands.py`

**Test targets:**
- `--dry-run` nie zapisuje chain do managera,
- pokazuje deterministic preview parsed chain,
- pokazuje warnings o legacy/inactive fields,
- exit code pozostaje przewidywalny.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
Expected: FAIL na braku `--dry-run`.

---

### Task 7: Implement `prompt-chain-apply --dry-run`

**Status:** functionally-complete / pyright-debt-followup

**Objective:** Pozwolić operatorowi zobaczyć, co zostanie zapisane, zanim dotknie storage.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Implementation targets:**
- parser flag `--dry-run`,
- parse + validate payload bez persistence,
- compact preview chain shape + warnings,
- opcjonalnie JSON reuse jeśli to redukuje duplikację.

**Constraint:**
Nie twórz nowego pipeline’u apply. Reuse istniejące parse/validate seams.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
- `ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`
- `.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`

---

## Phase 3 — Result consumption shortcuts

### Task 8: Add failing CLI tests for selective result extraction

**Status:** done

**Objective:** Zanim zmienisz run output modes, zamroź minimalne selektory wyniku.

**Files:**
- Modify: `tests/test_prompt_chain_cli.py`
- Modify: `cli/parser.py`
- Reference: `cli/commands.py`

**Test targets:**
- `--step-output <step_key>` drukuje tylko wskazany canonical output,
- `--step-alias <alias>` mapuje alias -> canonical key,
- `--final-step-meta` wypisuje tylko bounded terminal metadata,
- konflikty między selective output flags a `--json` / innymi trybami są jawnie blokowane.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
Expected: FAIL.

---

### Task 9: Implement selective result extraction flags

**Status:** done

**Objective:** Ograniczyć ręczne post-processing po run bez rozszerzania kontraktu backendowego.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Implementation targets:**
- `--step-output <canonical_key>`
- `--step-alias <alias>`
- `--final-step-meta`
- parser-level lub handler-level mutual exclusivity dla output modes

**Constraint:**
Nie zmieniaj shape `PromptChainRunResult`. To ma być consumption layer only.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
- `ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`
- `.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`

---

### Task 10: Add compact text result mode

**Status:** done

**Objective:** Dodać krótki operator-friendly handoff bez rezygnacji z pełnego outputu.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Implementation targets:**
- `--compact` dla prompt-chain-run,
- output ograniczony do: chain, status, final output preview, summary preview,
- spójność z `--output-file`.

**Constraint:**
Nie mnożyć równoległych formatów. Jeden bounded compact mode.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_cli.py -q`
- `ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py`

---

## Phase 4 — GUI editor ergonomics

### Task 11: Add failing GUI tests for step reorder controls

**Status:** done

**Objective:** Zamrozić prostą ergonomię reorder przed zmianą edytora.

**Files:**
- Modify: `tests/test_prompt_chain_dialog.py`
- Modify: `tests/test_prompt_chain_editor.py` if needed
- Reference: `gui/dialogs/prompt_chain_editor.py`

**Test targets:**
- move up / move down zmienia order_index deterministycznie,
- save zachowuje nową kolejność,
- UI nie wprowadza canvas/workflow-builder behavior.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_dialog.py -q`
Expected: FAIL.

---

### Task 12: Implement step reorder controls

**Status:** done

**Objective:** Ułatwić poprawianie chains bez ręcznego przepisywania kolejności.

**Files:**
- Modify: `gui/dialogs/prompt_chain_editor.py`
- Modify: `tests/test_prompt_chain_dialog.py`
- Optional: `tests/test_prompt_chain_editor.py`

**Implementation targets:**
- move step up,
- move step down,
- renumber order indexes after reorder,
- zachować czytelny, prosty layout.

**Constraint:**
Bez drag-and-drop jeśli wymaga dużego UI churn. Najpierw zwykłe przyciski.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_dialog.py -q`
- `ruff check gui/dialogs/prompt_chain_editor.py tests/test_prompt_chain_dialog.py`
- `.venv/bin/pyright gui/dialogs/prompt_chain_editor.py tests/test_prompt_chain_dialog.py`

---

### Task 13: Add failing GUI tests for duplicate-step and pre-save warnings

**Status:** done

**Objective:** Zamrozić małe UX conveniences przed implementacją.

**Files:**
- Modify: `tests/test_prompt_chain_dialog.py`
- Reference: `gui/dialogs/prompt_chain_editor.py`

**Test targets:**
- duplicate step tworzy nowy step z tym samym prompt reference i poprawnym order,
- pre-save warnings pokazują bounded issues,
- legacy/inactive semantics są oznaczane czytelnie.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_dialog.py -q`
Expected: FAIL.

---

### Task 14: Implement duplicate-step and pre-save warning summary

**Status:** done

**Objective:** Usprawnić edycję chainów bez budowania no-code toola.

**Files:**
- Modify: `gui/dialogs/prompt_chain_editor.py`
- Modify: `tests/test_prompt_chain_dialog.py`

**Implementation targets:**
- duplicate selected step,
- compact warning summary before accept/save,
- signal legacy/inactive semantics when imported payload je zawiera.

**Constraint:**
Warning summary ma być krótki i operator-facing; bez eksperckiego debug panelu.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_dialog.py -q`
- `ruff check gui/dialogs/prompt_chain_editor.py tests/test_prompt_chain_dialog.py`

---

## Phase 5 — Persisted lightweight history

### Task 15: Document the minimum durable chain-run evidence record

**Status:** done

**Objective:** Zanim dodasz persistence, ustal minimalny bounded record.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: active execution ledger for this plan
- Reference: repository/history seams

**Record target:**
- chain id,
- chain name snapshot or resolvable id,
- run timestamp,
- aggregate status,
- input preview,
- final output preview,
- final step output key.

**Verification:**
- Re-read changed section and confirm no analytics-first expansion.

---

### Task 16: Add failing tests for persisted recent chain history

**Status:** done

**Objective:** Zamrozić minimum trwałej historii przed implementacją storage.

**Files:**
- Modify: `tests/test_prompt_chain_backend.py`
- Modify: repository/history test file once seam is identified
- Reference: `core/prompt_manager/chains.py`

**Test targets:**
- successful run zapisuje recent history entry,
- history jest bounded,
- newest-first retrieval działa,
- brak analytics/dashboard semantics.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_backend.py -q`
Expected: FAIL.

---

### Task 17: Implement persisted recent chain history backend

**Status:** done

**Objective:** Dodać trwałe evidence surface dla trust/debug/reuse.

**Files:**
- Modify: repository/history layer after seam discovery
- Modify: `core/prompt_manager/chains.py`
- Modify: `tests/test_prompt_chain_backend.py`
- Optional docs sync: `docs/README-DEV.md`

**Implementation targets:**
- write bounded history record after run,
- retrieval newest-first,
- bounded retention,
- zero dashboard expansion.

**Constraint:**
Nie archiwizować pełnych request/response blobs domyślnie.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_backend.py -q`
- `ruff check core/prompt_manager/chains.py tests/test_prompt_chain_backend.py`
- `.venv/bin/pyright core/prompt_manager/chains.py tests/test_prompt_chain_backend.py`

---

### Task 18: Surface persisted history in GUI and optional CLI inspect

**Status:** pending

**Objective:** Uczynić trwałą historię realnie użyteczną operatorowi.

**Files:**
- Modify: `gui/dialogs/prompt_chains.py`
- Modify: `cli/commands.py` if adding lightweight CLI inspect surface
- Modify: `tests/test_prompt_chain_dialog.py`
- Modify: `tests/test_prompt_chain_cli.py` if needed

**Implementation targets:**
- GUI: recent runs for selected chain, bounded and readable,
- optionally CLI: `prompt-chain-history` lub mały history section w show,
- evidence-only surface: timestamp, status, previews.

**Constraint:**
Jeśli CLI path robi się za szeroki, dowieźć najpierw GUI inspect surface i zostawić CLI jako next candidate.

**Verification:**
Run:
- `pytest tests/test_prompt_chain_dialog.py tests/test_prompt_chain_cli.py -q`
- `ruff check gui/dialogs/prompt_chains.py cli/commands.py tests/test_prompt_chain_dialog.py tests/test_prompt_chain_cli.py`

---

## Current recommended next slice

**Task 18** jako następny bounded cycle.

Powód:
- backend persisted recent history jest już gotowy, ale jeszcze nie jest wystawiony operatorowi,
- to domyka praktyczny feedback loop dla trust/debug/reuse bez rozbudowy dashboard semantics,
- GUI inspect surface jest zgodny z celem cyklu i może pozostać bounded oraz readable,
- CLI path powinien pozostać opcjonalny i wejść tylko jeśli nie poszerza scope ponad mały inspect/history surface.

---

## Update workflow after each implemented slice

Po każdym zakończonym tasku lub małym task bundle:
1. zaktualizuj ten plan (`Status`, `Implemented`, `Verified`),
2. jeśli zmieniła się feature truth — zaktualizuj `docs/plans/2026-05-06-prompt-chain-ssot.md`,
3. sprawdź, czy starsze prompt-chain ledger files nie wskazują nadal starego next slice,
4. dopiero potem przejdź do kolejnego slice’a.

---

## Definition of done

Slice jest skończony tylko wtedy, gdy:
- nie poszerza prompt chains poza bounded linear runner,
- core/CLI/GUI pozostają spójne semantycznie,
- focused tests przechodzą,
- Ruff przechodzi,
- Pyright na dotkniętych plikach przechodzi lub jest jawnie oznaczony jako `do weryfikacji`,
- docs/ledger state nie zostawia sprzecznych next-step signals.
