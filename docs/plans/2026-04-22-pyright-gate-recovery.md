# Pyright Gate Recovery Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Utrzymać zielony quality gate na GitHubie, urealnić plan względem stanu repo i przygotować bezpieczny następny etap rozszerzania typed coverage bez rozwalania CI.

**Architecture:** Część recovery już została wykonana: GH gate jest zielony po zawężeniu scope do `config`. Plan nie powinien już zakładać pracy „od zera”, tylko rozdzielać stan osiągnięty od kolejnych kroków. Następny sensowny etap to domknięcie `main.py`, tak aby można było rozszerzyć gate z `config` do `main.py + config`, a dopiero potem przechodzić do większych obszarów.

**Tech Stack:** GitHub Actions, Pyright 1.1.408, Python 3.13, repo PromptManager.

---

## Current verified status (2026-04-23, refreshed)

### Potwierdzone
- repo: `/home/voytas/projects/PromptManager`
- branch: `master`
- working tree: czyste
- remote: `https://github.com/voytas75/PromptManager.git`
- `HEAD`: `0898558` — `fix: preserve embedding model from dotenv config`
- `pyrightconfig.json` **nie** ma już `include: ["."]`; obecny include to:
  - `main.py`
  - `core`
  - `config`
  - `gui`
  - `models`
  - `tests`
- `.github/workflows/quality-gates.yml` uruchamia dziś:

```yaml
- name: Pyright
  run: .venv/bin/pyright main.py config
```

- `docs/README-DEV.md` jest już urealnione do aktualnego gate'u `pyright main.py config`.
- commity związane z tym etapem już istnieją:
  - `2f15f0b` — `fix: type main entrypoint for pyright gate expansion`
  - `503e546` — `ci: expand pyright gate and align docs`
- lokalna weryfikacja wykonana 2026-04-23 na tym środowisku potwierdza `pyright main.py config`:

```text
0 errors, 0 warnings, 0 informations
```

- potwierdzony GH run po rozszerzeniu gate:
  - run id: `24840076043`
  - status: `success`
  - commit: `503e546` — `ci: expand pyright gate and align docs`
  - job `quality`: wszystkie kroki zielone, w tym `Pyright`, `Pytest` i `Ensure clean tree`
- potwierdzony GH run po fixie precedence dla `PROMPT_MANAGER_EMBEDDING_MODEL` z dotenv:
  - run id: `24847766599`
  - status: `success`
  - commit: `0898558` — `fix: preserve embedding model from dotenv config`
  - job `quality`: wszystkie kroki zielone, w tym `Ruff (autofix)`, `Ruff (format)`, `Ruff (verify)`, `Pyright`, `Pytest` i `Ensure clean tree`

### Nadal otwarte
- roadmapa dalszego rozszerzania strict coverage **już istnieje** w `docs/plans/pyright-strict-expansion-roadmap.md`; trzeba ją teraz traktować jako aktywny plan kolejnych etapów, nie jako brakujący artefakt.
- regres dla `PROMPT_MANAGER_ENV_FILE` / dotenv alias precedence został już częściowo domknięty testem dla `PROMPT_MANAGER_EMBEDDING_MODEL`, ale warto dopisać osobne przypadki dla pozostałych canonical pól LiteLLM i alias fallbacków, żeby nie wrócił podobny bug w innej gałęzi precedence.
- podczas uruchamiania GUI analytics panel wykonuje startowy probe embeddingów przez `manager.diagnose_embeddings()`. Na tym środowisku aktywna konfiguracja LiteLLM dla embeddings wymaga modelu w formacie deployment-aware dla Azure (np. `azure/UDTEMBED3L`); pozostawienie samego `text-embedding-3-large` kończy się błędem backendu i powoduje trzy banery LiteLLM przy starcie okna. To nie jest problem samego GUI, tylko hałaśliwe ujawnienie błędnej nazwy modelu/deploymentu dla embeddingów.

### Historyczny zestaw błędów w `main.py` z momentu tworzenia planu

```text
/home/voytas/projects/PromptManager/main.py:25:24 - error: Import "PromptManagerSettings" is not accessed (reportUnusedImport)
/home/voytas/projects/PromptManager/main.py:25:47 - error: Import "SettingsError" is not accessed (reportUnusedImport)
/home/voytas/projects/PromptManager/main.py:41:30 - error: Type of "run_default_mode" is partially unknown
/home/voytas/projects/PromptManager/main.py:61:5 - error: Function "_setup_logging" is not accessed (reportUnusedFunction)
/home/voytas/projects/PromptManager/main.py:61:20 - error: Type of parameter "config_path" is unknown (reportUnknownParameterType)
/home/voytas/projects/PromptManager/main.py:61:20 - error: Type annotation is missing for parameter "config_path" (reportMissingParameterType)
/home/voytas/projects/PromptManager/main.py:63:28 - error: Argument type is unknown (reportUnknownArgumentType)
```

---

## Reframed plan

### Done already

#### Completed A: Zawęź GH gate do stabilnego scope

**Status:** completed

**What changed:**
- workflow nie uruchamia już pełnego `pyright`,
- aktywny gate został najpierw zawężony do `pyright config`,
- ten etap odblokował zielone GH Actions i pozwolił przejść do następnego rozszerzenia.

**Evidence:**
- historyczne commity `82cc905`, `f6eceb0`

#### Completed B: Usuń `include: ["."]` z `pyrightconfig.json`

**Status:** completed

**What changed:**
- problem z niekontrolowanym zakresem skanowania został usunięty,
- performance issue nie wrócił.

**Evidence:**
- `pyrightconfig.json` ma jawny include list zamiast `.`

#### Completed C: Domknij `main.py` i rozszerz gate do `main.py config`

**Status:** completed in repo, local re-verification pending

**What changed:**
- `main.py` został poprawiony pod gate expansion,
- workflow uruchamia już `pyright main.py config`,
- `docs/README-DEV.md` opisuje obecny scope zgodny z CI.

**Evidence:**
- commit `2f15f0b` — `fix: type main entrypoint for pyright gate expansion`
- commit `503e546` — `ci: expand pyright gate and align docs`

---

### Task 1: Potwierdź aktualny stan gate na żywym checkoutcie

**Objective:** Zweryfikować lokalnie i na GitHubie, że etap `main.py + config` naprawdę jest domknięty na bieżącym stanie repo.

**Files:**
- Verify: `.github/workflows/quality-gates.yml`
- Verify: `docs/README-DEV.md`
- Verify: `main.py`

**Step 1: Odtwórz środowisko lokalne**

Run:

```bash
python3 -m ensurepip --upgrade
python3 -m venv .venv
. .venv/bin/activate
python -m ensurepip --upgrade
pip install -e .[dev]
```

**Step 2: Uruchom faktyczny gate lokalnie**

Run:

```bash
.venv/bin/pyright main.py config
```

**Expected:**

```text
0 errors, 0 warnings, 0 informations
```

Jeśli wynik nie jest zielony, zaktualizuj ten plan o realny zestaw błędów zamiast ufać historycznym wpisom.

**Step 3: Potwierdź GH Actions po właściwych commitach**

Run:

```bash
gh run list --repo voytas75/PromptManager --limit 5
gh run view --repo voytas75/PromptManager <run_id>
```

Potwierdź run dla commitów `2f15f0b` / `503e546` lub nowszych.

---

### Task 2: Dodać trwały test regresyjny dla `PROMPT_MANAGER_ENV_FILE`

**Objective:** Zapisać w repo rzeczywisty przypadek precedence dla dotenv/config path bez polegania na brudnym stanie środowiska.

**Files:**
- Create or modify: `tests/test_settings_env_file_regression.py` albo `tests/test_settings.py`
- Verify: `config/settings.py`

**Observed blocker from refresh:**
- na tym środowisku istnieją ambient alias env vars (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL`),
- próba szybkiego testu mieszała dwa problemy naraz: precedence dotenv vs ambient env,
- dodatkowo warto unikać literalów sekretów, które lokalnie bywają maskowane przy automatycznej edycji.

**Recommended approach:**
1. zbudować test na wartościach niesekretnych (`LITELLM_API_VERSION`, kontrolowany `LITELLM_API_BASE` albo inne bezpieczne pole),
2. jawnie wyczyścić alias env vars używane przez loader,
3. najpierw uruchomić sam test,
4. potem `ruff` i `pyright` dla dotkniętych plików,
5. dopiero wtedy commit.

---

### Task 3: Utrzymuj i aktualizuj roadmapę dalszego rozszerzania strict coverage

**Status:** artifact already created, maintain forward

**Objective:** Po odblokowaniu `main.py + config` utrzymać kontrolowany plan zdejmowania długu typów i prowadzić kolejne etapy z jednego aktywnego pliku roadmapy.

**Files:**
- Maintain: `docs/plans/pyright-strict-expansion-roadmap.md`

**Current status:**
- plik roadmapy już istnieje,
- obecny baseline zapisany w roadmapie to `pyright main.py config`,
- następny kandydat scope to `models`.

**Maintenance rules:**
1. aktualizuj roadmapę po każdym realnym rozszerzeniu scope,
2. zapisuj w niej potwierdzony lokalny wynik i powiązany GH run,
3. nie oznaczaj fazy jako zakończonej bez zielonego CI,
4. trzymaj roadmapę zgodną z `docs/README-DEV.md` i workflow.

**Suggested success metric per phase:**
- lokalny Pyright dla danego scope = zielony,
- workflow scope rozszerzony dopiero po lokalnym potwierdzeniu,
- GH run zielony po każdym etapie,
- dokumentacja zaktualizowana przy każdej zmianie scope.

---

### Task 4: Ogranicz hałas LiteLLM z nieudanego embedding probe przy starcie GUI

**Objective:** Wyciszyć tylko znany, konkretny przypadek startowego probe embeddingów (brak zasobu/deploymentu Azure dla embeddings) bez maskowania innych błędów LiteLLM.

**Files:**
- Modify: `gui/analytics_panel.py`
- Verify: `core/analytics_dashboard.py`
- Verify: `core/prompt_manager/analytics.py`
- Add tests near analytics GUI / dashboard coverage if brak odpowiedniego testu

**Observed root cause:**
- `AnalyticsDashboardPanel.__init__()` robi `self.refresh()` od razu przy budowie GUI,
- `refresh()` woła `build_analytics_snapshot(...)`,
- snapshot woła `manager.diagnose_embeddings()`,
- przy aktywnym `embedding_backend=litellm` model embeddingów dla Azure musi wskazywać deployment-aware identyfikator LiteLLM (np. `azure/UDTEMBED3L`),
- pozostawienie ogólnej nazwy modelu (`text-embedding-3-large`) prowadzi do błędu backendu i LiteLLM drukuje banner do stdout,
- obecny efekt uboczny: użytkownik widzi trzy banery LiteLLM przy samym starcie okna, mimo że nie uruchamiał ręcznie diagnostyki.

**Constraint:**
- nie maskować globalnie LiteLLM,
- nie ukrywać innych błędów runtime,
- zawęzić zmianę tylko do tego jednego startowego probe / znanego przypadku 404 resource-not-found dla embeddings.

**Recommended direction:**
1. traktować startowy embedding probe w analytics jako **best-effort**,
2. dodać wąskie rozpoznanie błędu tylko dla embedding diagnostics startup path,
3. zamiast dopuszczać banner LiteLLM do stdout przy tym jednym przypadku, zamienić go na spokojny status w UI, np. `Embedding backend unavailable: Azure resource/deployment not found`,
4. pozostawić wszystkie inne wyjątki bez takiego specjalnego tłumienia.

**Preferred implementation shape:**
- najwęższe miejsce to `core/prompt_manager/analytics.py::diagnose_embeddings()` albo wyłącznie ścieżka wywołania z `gui/analytics_panel.py`,
- rozpoznawać tylko błędy odpowiadające:
  - `404 Resource not found`,
  - `DeploymentNotFound`,
  - wyłącznie dla embedding diagnostics probe,
- wynik zapisywać do `backend_ok=False` / `backend_message=...`,
- nie propagować tego przypadku dalej jako hałaśliwy banner startowy.

**Do not do:**
- nie ustawiaj globalnego `litellm.suppress_debug_info=True` dla całej aplikacji,
- nie łap szeroko wszystkich `Exception` z LiteLLM i nie zamieniaj ich na `pass`,
- nie wyłączaj całkiem `diagnose_embeddings()` dla wszystkich scenariuszy bez decyzji produktowej.

**Verification target:**
- przy znanym braku embedding resource GUI startuje bez trzech bannerów LiteLLM,
- analytics dalej pokazuje czytelny stan `embedding backend unavailable`,
- inny błąd LiteLLM poza tym przypadkiem nadal jest widoczny i nie zostaje zamaskowany.

---

### Task 5: Wypchnij i potwierdź GH run po odświeżeniu planu lub kolejnych zmianach

**Objective:** Zweryfikować, że repo i CI nadal są zgodne z odświeżonym planem.

**Files:**
- No code changes required

**Step 1: Push**

```bash
git push origin master
```

**Step 2: Sprawdź GH Actions**

Run:

```bash
gh run list --repo voytas75/PromptManager --limit 3
gh run view --repo voytas75/PromptManager <run_id>
```

**Expected:**
- `Pyright` nadal kończy się szybko,
- pipeline pozostaje zielony,
- brak regresji do szerokiego scope,
- dla lokalnego HEAD po commicie `0929d38` istnieje osobny potwierdzony run po pushu.

---

## Verification checklist

- [x] `pyrightconfig.json` nie ma już `include: ["."]`
- [x] `.github/workflows/quality-gates.yml` używa zawężonego scope Pyrighta
- [x] lokalnie potwierdzono bieżące `./.venv/bin/pyright main.py config` na tym checkoutcie
- [x] potwierdzono aktualny GH run dla commitów po rozszerzeniu gate
- [ ] istnieje trwały test regresyjny dla `PROMPT_MANAGER_ENV_FILE` / dotenv precedence
- [x] istnieje zapisany plan dalszego rozszerzania strict-checka

---

## Notes / guardrails

- Nie wracaj do `include: ["."]`.
- Nie wyłączaj Pyrighta z CI.
- Nie rozszerzaj workflow na kolejny obszar, dopóki lokalny scope nie jest zielony.
- Nie zostawiaj w dokumentacji deklaracji szerszych niż realny CI.
- Najbliższy sensowny krok to **naprawa `main.py`**, nie ruszanie od razu `core`, `gui`, `models` ani `tests`.
