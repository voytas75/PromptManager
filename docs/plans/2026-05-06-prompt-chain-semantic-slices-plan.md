# Prompt Chain Semantic Slices Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Domknąć semantykę prompt chain w PromptManager bez poszerzania scope poza bounded linear runner.

**Architecture:** Plan rozdziela pracę na małe slices wokół jednego SSOT kontraktu backendowego. Najpierw ujednolicamy status runu i semantykę final-vs-terminal step w backendzie, potem przepinamy CLI i GUI na ten kontrakt, a dopiero później czyścimy telemetry/history i legacy-runtime ergonomics.

**Tech Stack:** Python 3.13+, dataclasses, PromptManager backend mixins, CLI argparse commands, Qt GUI dialogs, pytest, ruff, pyright.

---

## Scope i zasady

- Nie dodawać branching, loopów, retry workflow semantics ani scheduler-first behavior.
- Preferować jeden backendowy kontrakt używany przez CLI i GUI.
- Każdy slice ma być samowystarczalny i weryfikowalny osobno.
- Najpierw RED testy, potem minimalna implementacja, potem bounded docs sync.
- Nie przepisywać szeroko istniejącego modelu, jeśli wystarczy warstwa kontraktu/adaptacji.

---

## Slice A — Run status + final vs terminal semantics

### Task A1: Spisać kontrakt run status i final/terminal step w SSOT

**Objective:** Ustalić jednoznaczną definicję `run_status`, `final_step_*` i `terminal_step_*` przed zmianą kodu.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`

**Steps:**
1. Dopisać do SSOT definicje:
   - `run_status`
   - `final_step_*` = ostatni udany krok dający finalny output
   - `terminal_step_*` = ostatni wykonany krok niezależnie od sukcesu
2. Dopisać do rollout plan nowy slice semantyczny jako planned.
3. Zweryfikować, że zapis nie wprowadza workflow-engine semantics.

**Verification:**
- `python -m pytest tests/test_prompt_chain_backend.py -q` nie jest wymagane na tym kroku, ale plik planu/SSOT musi być spójny tekstowo.

**Commit:**
- `git commit -m "docs: define prompt chain run and terminal step semantics"`

---

### Task A2: Dodać RED testy backendowe dla run status i terminal step

**Objective:** Złapać obecny semantyczny drift zanim ruszy implementacja.

**Files:**
- Modify: `tests/test_prompt_chain_backend.py`

**Step 1: Write failing tests**

Dodać testy dla scenariuszy:
- sukces wszystkich kroków → `run_status == success`
- wcześniejszy sukces + późniejszy fail z `stop_on_failure=False` → `run_status == partial_success`, `final_step_*` wskazuje ostatni sukces, `terminal_step_*` wskazuje ostatni wykonany fail
- brak sukcesu + fail → `run_status == failed`
- skip bez fail i bez final output → `run_status == skipped`

**Step 2: Run tests to verify failure**

Run:
```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py -q
```
Expected: FAIL w nowych testach kontraktu.

**Step 3: Minimal test harness**

Jeśli obecny harness nie pozwala łatwo wymusić mixed outcomes, dodać mały stub manager/executor tylko dla tych przypadków.

**Step 4: Re-run targeted tests**

Run:
```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py -q
```
Expected: nadal FAIL, ale tylko na brakującej implementacji.

**Commit:**
- brak commitu na RED-only jeśli od razu przechodzimy do A3.

---

### Task A3: Wdrożyć backendowy helper `run_status` i `terminal_step_*`

**Objective:** Przenieść całą semantykę run outcome do backendu jako jedno źródło prawdy.

**Files:**
- Modify: `core/prompt_manager/chains.py`
- Test: `tests/test_prompt_chain_backend.py`

**Implementation direction:**
- Dodać helper typu `_compute_chain_run_status(step_runs, final_output_text)`.
- Rozszerzyć `PromptChainRunResult` o:
  - `run_status`
  - `terminal_step_id`
  - `terminal_step_label`
  - `terminal_step_output_key`
  - `terminal_step_status`
- Upewnić się, że `final_step_*` i `terminal_step_*` są wyliczane niezależnie i spójnie.

**Step 1: Implement minimal code**
- Najpierw wypełnić nowe pola podczas `run_prompt_chain()`.
- Następnie przepiąć `_build_chain_history_record()` na `result.run_status` zamiast własnej heurystyki.

**Step 2: Run tests**

Run:
```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py -q
.venv/bin/ruff check core/prompt_manager/chains.py tests/test_prompt_chain_backend.py
.venv/bin/pyright core/prompt_manager/chains.py tests/test_prompt_chain_backend.py
```
Expected: PASS.

**Commit:**
- `git commit -m "feat: unify prompt chain run status semantics"`

---

### Task A4: Przepiąć CLI na backendowy kontrakt semantyczny

**Objective:** Usunąć rozjazd między backendem a `prompt-chain-run`.

**Files:**
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Step 1: Write failing tests**
- Dodać test `--status-only` czyta `result.run_status`.
- Dodać test `--final-step-meta` nie miesza statusu terminal step z final step.
- Dodać test JSON payload zawiera nowe pola terminalne, jeśli expose jest zatwierdzony.

**Step 2: Run tests to verify failure**

Run:
```bash
.venv/bin/pytest tests/test_prompt_chain_cli.py -q
```
Expected: FAIL.

**Step 3: Implement minimal code**
- `final_status` usunąć jako lokalną heurystykę tam, gdzie może przyjść z `result.run_status`.
- `--final-step-meta`:
  - albo zwracać tylko final-step semantics,
  - albo rozszerzyć payload o `terminal_step_*` i `run_status`.
- `_build_prompt_chain_run_json_payload()` zsynchronizować z nowym kontraktem.

**Step 4: Verify**

Run:
```bash
.venv/bin/pytest tests/test_prompt_chain_cli.py -q
.venv/bin/ruff check cli/commands.py tests/test_prompt_chain_cli.py
.venv/bin/pyright cli/commands.py tests/test_prompt_chain_cli.py
```
Expected: PASS.

**Commit:**
- `git commit -m "feat: align prompt chain cli with backend run semantics"`

---

## Slice B — Recent history as one SSOT

### Task B1: Dodać RED testy dla history status i limit normalization

**Objective:** Zablokować regresje bounded history contract zanim GUI/CLI zaczną z niego czytać.

**Files:**
- Modify: `tests/test_prompt_chain_backend.py`

**Tests to add:**
- `list_recent_prompt_chain_runs(limit=0)` zwraca co najmniej 1 lub dokumentowany fallback
- `run_status` w recordzie bierze się z backendowego kontraktu, nie z lokalnej heurystyki
- history newest-first dalej działa po zmianie struktury result

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py -q
```
Expected: FAIL przed implementacją jeśli potrzeba.

---

### Task B2: Ujednolicić backend history record i GUI consumer

**Objective:** GUI ma renderować to samo, co backend uważa za recent runs.

**Files:**
- Modify: `core/prompt_manager/chains.py`
- Modify: `gui/dialogs/prompt_chains.py`
- Modify: `tests/test_prompt_chain_dialog.py`

**Implementation direction:**
- Nie liczyć statusu historii osobno w GUI.
- GUI powinno użyć backendowego recent history seam albo helpera wspólnego z backendem.
- Ustalić jedną retencję dla session-level history albo jasno opisać różnicę UI vs backend.

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_dialog.py -q
.venv/bin/ruff check core/prompt_manager/chains.py gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
.venv/bin/pyright core/prompt_manager/chains.py gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
```

**Commit:**
- `git commit -m "feat: unify prompt chain recent history semantics"`

---

### Task B3: Dodać CLI read-only subfunction `prompt-chain-run --recent`

**Objective:** Udostępnić recent runs bez GUI i bez rozbudowy dashboardu.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`

**Behavior:**
- `prompt-chain-run --recent` wypisuje bounded recent runs.
- tryb tekstowy: timestamp, chain, status, input preview, output preview
- opcjonalnie `--json` jeśli nie koliduje z obecną strukturą komendy

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_cli.py -q
.venv/bin/ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
```

**Commit:**
- `git commit -m "feat: add prompt chain recent runs cli view"`

---

## Slice C — Web search telemetry correctness

### Task C1: Spisać docelową telemetrię enrichment

**Objective:** Zastąpić niejawne zgadywanie jawnie opisanym kontraktem.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`

**Semantics to define:**
- `web_search_requested` = operator/runtime requested it
- `web_search_applied` = enrichment layer potwierdził modyfikację używając jawnego wyniku, nie heurystyki tekstowej
- opcjonalnie `web_search_note` / `skip_reason` dla unavailable/no-result

---

### Task C2: Dodać RED testy dla explicit enrichment result

**Objective:** Wymusić zmianę sygnatury/helpera enrichment zamiast dalszego porównywania stringów.

**Files:**
- Modify: `tests/test_prompt_chain_backend.py`

**Tests to add:**
- enrichment requested + applied true
- enrichment requested + applied false despite successful call
- enrichment unavailable → `skip_reason` / note

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py -q
```
Expected: FAIL.

---

### Task C3: Wdrożyć explicit enrichment result

**Objective:** Zmienić telemetry contract na potwierdzony, nie heurystyczny.

**Files:**
- Modify: `core/prompt_manager/chains.py`
- Test: `tests/test_prompt_chain_backend.py`

**Implementation direction:**
- `_maybe_enrich_with_web_search()` powinno zwracać strukturę albo tuple z `text`, `applied`, opcjonalnie `reason`.
- `PromptChainStepRun` ma dostawać te pola z tego źródła.

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py -q
.venv/bin/ruff check core/prompt_manager/chains.py tests/test_prompt_chain_backend.py
.venv/bin/pyright core/prompt_manager/chains.py tests/test_prompt_chain_backend.py
```

**Commit:**
- `git commit -m "feat: make prompt chain web search telemetry explicit"`

---

## Slice D — Runtime-only clarity for legacy fields

### Task D1: Doprecyzować w SSOT status legacy runtime fields

**Objective:** Ustalić, że `input_template`, `condition`, `variables_schema` są compatibility-boundary fields, nie aktywna semantyka runtime.

**Files:**
- Modify: `docs/plans/2026-05-06-prompt-chain-ssot.md`
- Modify: `docs/product-ssot.md`

---

### Task D2: Dodać CLI subfunction `prompt-chain-show --runtime-semantics`

**Objective:** Pozwolić operatorowi szybko zobaczyć aktywne vs legacy semantics.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Behavior:**
- prosty output tekstowy/JSON:
  - active runtime semantics
  - compatibility-only fields detected

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_cli.py -q
.venv/bin/ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
```

**Commit:**
- `git commit -m "feat: add prompt chain runtime semantics view"`

---

### Task D3: Dodać `prompt-chain-validate --strict-runtime`

**Objective:** Ostrzegać przed payloadami sugerującymi nieaktywną semantykę runtime.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Behavior:**
- normal validate: jak dziś
- strict-runtime: ostrzejsze warningi / non-zero exit tylko jeśli użytkownik wyraźnie tego chce

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_cli.py -q
.venv/bin/ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
```

**Commit:**
- `git commit -m "feat: add strict runtime prompt chain validation"`

---

### Task D4: Dodać `prompt-chain-export --runtime-only`

**Objective:** Ułatwić clean export bez legacy payload noise.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_cli.py -q
.venv/bin/ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
```

**Commit:**
- `git commit -m "feat: add runtime-only prompt chain export"`

---

## Slice E — Small operator ergonomics

### Task E1: CLI `--artifact-dir` for run outputs

**Objective:** Ułatwić seryjne zapisywanie run artifacts bez ręcznego wymyślania nazw plików.

**Files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Modify: `tests/test_prompt_chain_cli.py`

**Behavior:**
- jeśli podano `--artifact-dir`, zapisywać artifact pod deterministyczną nazwą zawierającą timestamp + chain id/name
- nie pozwalać łączyć bez sensu z `--output-file`, jeśli to konflikt

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_cli.py -q
.venv/bin/ruff check cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
.venv/bin/pyright cli/parser.py cli/commands.py tests/test_prompt_chain_cli.py
```

**Commit:**
- `git commit -m "feat: add prompt chain artifact directory output"`

---

### Task E2: GUI copy conveniences

**Objective:** Zmniejszyć ręczne kopiowanie final output / summary / output keys.

**Files:**
- Modify: `gui/dialogs/prompt_chains.py`
- Modify: `tests/test_prompt_chain_dialog.py`

**Features:**
- Copy final output
- Copy final summary
- Copy selected step output key

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_dialog.py -q
.venv/bin/ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
.venv/bin/pyright gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
```

**Commit:**
- `git commit -m "feat: add prompt chain gui copy actions"`

---

### Task E3: GUI rerun with previous input

**Objective:** Przyspieszyć iteracyjne użycie chainów bez nowej semantyki workflow.

**Files:**
- Modify: `gui/dialogs/prompt_chains.py`
- Modify: `tests/test_prompt_chain_dialog.py`

**Behavior:**
- po runie zachować ostatni input dla wybranego chaina
- dodać action/button do ponownego uruchomienia na tym samym input bez ręcznego przepisywania

**Verification:**
```bash
.venv/bin/pytest tests/test_prompt_chain_dialog.py -q
.venv/bin/ruff check gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
.venv/bin/pyright gui/dialogs/prompt_chains.py tests/test_prompt_chain_dialog.py
```

**Commit:**
- `git commit -m "feat: add prompt chain rerun convenience"`

---

## Recommended execution order

1. Slice A — najwyższy priorytet, bo domyka semantykę rdzenia.
2. Slice B — dopiero po A, bo history ma konsumować finalny kontrakt.
3. Slice C — po A/B, bo telemetry ma siąść na stabilnym result model.
4. Slice D — po ustabilizowaniu runtime semantics.
5. Slice E — na końcu jako bounded UX polish.

---

## Minimum verification gate per slice

Dla każdego slice przed commitem uruchomić:

```bash
.venv/bin/ruff check <modified_files>
.venv/bin/pyright <modified_files>
.venv/bin/pytest <targeted_tests>
```

Po większych zmianach backend+CLI/GUI dodatkowo:

```bash
.venv/bin/pytest tests/test_prompt_chain_backend.py tests/test_prompt_chain_cli.py tests/test_prompt_chain_dialog.py -q
```

---

## Done criteria

Slice uznajemy za zamknięty tylko gdy:
- kontrakt jest opisany w SSOT lub rollout ledgerze,
- RED testy zostały dodane i przeszły po implementacji,
- CLI/GUI nie liczą już lokalnie sprzecznych semantyk, jeśli slice dotyczy wspólnego kontraktu,
- brak nowego scope creep w stronę workflow engine,
- targeted ruff + pyright + pytest są zielone.
