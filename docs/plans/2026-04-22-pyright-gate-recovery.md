# Pyright Gate Recovery Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Utrzymać zielony quality gate na GitHubie, urealnić plan względem stanu repo i przygotować bezpieczny następny etap rozszerzania typed coverage bez rozwalania CI.

**Architecture:** Część recovery już została wykonana: GH gate jest zielony po zawężeniu scope do `config`. Plan nie powinien już zakładać pracy „od zera”, tylko rozdzielać stan osiągnięty od kolejnych kroków. Następny sensowny etap to domknięcie `main.py`, tak aby można było rozszerzyć gate z `config` do `main.py + config`, a dopiero potem przechodzić do większych obszarów.

**Tech Stack:** GitHub Actions, Pyright 1.1.408, Python 3.13, repo PromptManager.

---

## Current verified status (2026-04-23)

### Potwierdzone
- repo: `/home/voytas/projects/PromptManager`
- branch: `master`
- working tree: czyste
- remote: `https://github.com/voytas75/PromptManager.git`
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
  run: .venv/bin/pyright config
```

- lokalnie `./.venv/bin/pyright config` przechodzi:

```text
0 errors, 0 warnings, 0 informations
```

- ostatni udany GH run:
  - run id: `24804265867`
  - status: `success`
  - commit: `docs: add product direction SSOT`
- `docs/README-DEV.md` zostało już częściowo urealnione i opisuje, że obecny gate dotyczy `pyright config`.

### Nadal otwarte
- `./.venv/bin/pyright main.py config` nadal failuje na `main.py` z **7 błędami**.
- dokumentacja jest jeszcze niespójna wewnętrznie:
  - u góry mówi o gate `pyright config`,
  - niżej nadal zawiera starsze sformułowanie typu „`pyright` must pass with zero warnings”.
- nie ma jeszcze osobnego pliku roadmapy rozszerzania strict coverage.

### Aktualny zestaw błędów w `main.py`

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
- aktywny gate to obecnie `pyright config`,
- GH Actions jest znowu zielone i szybkie.

**Evidence:**
- `.github/workflows/quality-gates.yml`
- GH run `24804265867` = success

#### Completed B: Usuń `include: ["."]` z `pyrightconfig.json`

**Status:** completed

**What changed:**
- problem z niekontrolowanym zakresem skanowania został usunięty,
- performance issue nie wrócił.

**Evidence:**
- `pyrightconfig.json` ma jawny include list zamiast `.`

---

### Task 1: Domknij `main.py`, żeby przygotować rozszerzenie gate

**Objective:** Doprowadzić `pyright main.py config` do zielonego stanu lokalnie bez naruszania obecnie działającego CI.

**Files:**
- Modify: `main.py`
- Verify: `config/`

**Step 1: Usuń nieużywane importy w `main.py`**

Usuń lub przebuduj import fallbackowy tak, by Pyright nie widział nieużywanych symboli:
- `PromptManagerSettings`
- `SettingsError`

Najpierw sprawdź, czy te symbole naprawdę są potrzebne runtime’owo. Jeśli służą tylko kompatybilności testów/stubów, uprość blok `try/except` tak, żeby nie zostawiać martwych importów.

**Step 2: Daj jawny typ helperowi `_setup_logging`**

Obecnie:

```python
def _setup_logging(config_path) -> None:
    _runtime_setup_logging(config_path)
```

Docelowy minimalny kierunek:

```python
from pathlib import Path

def _setup_logging(config_path: str | Path) -> None:
    _runtime_setup_logging(config_path)
```

Jeśli `_runtime_setup_logging` oczekuje węższego typu, dopasuj anotację do realnej sygnatury.

**Step 3: Rozwiąż `run_default_mode` partially unknown**

Najpierw przeczytaj definicję `cli.gui_launcher.run_default_mode` i popraw typy u źródła, jeśli tam brakuje jawnych anotacji.

Preferowana kolejność:
1. popraw sygnaturę w `cli/gui_launcher.py`,
2. jeśli trzeba, doprecyzuj importowane typy (`PromptManager`, settings, argparse namespace),
3. unikaj lokalnych obejść typu `cast(...)` bez potrzeby.

**Step 4: Rozwiąż `reportUnusedFunction` dla `_setup_logging`**

Jeżeli wrapper `_setup_logging` jest potrzebny tylko dla kompatybilności testów/legacy entry points, sprawdź testy i użycia.

Możliwe bezpieczne opcje:
- jeśli istnieje rzeczywiste użycie poza runtime, zostaw wrapper i udokumentuj/oznacz go tak, by Pyright nie traktował go jako martwego kodu tylko wtedy, gdy to uzasadnione,
- jeśli nie ma już żadnego użycia, usuń wrapper i zaktualizuj testy.

Nie wyciszaj ostrzeżenia „na ślepo”. Najpierw potwierdź potrzebę wrappera.

**Step 5: Zweryfikuj lokalnie**

Run:

```bash
.venv/bin/pyright main.py config
```

Expected:

```text
0 errors, 0 warnings, 0 informations
```

**Step 6: Commit**

```bash
git add main.py cli/gui_launcher.py config
git commit -m "fix: type main entrypoint for pyright gate expansion"
```

---

### Task 2: Rozszerz GH gate z `config` do `main.py config`

**Objective:** Podnieść wartość typed smoke gate bez wracania do szerokiego czerwonego scope.

**Files:**
- Modify: `.github/workflows/quality-gates.yml`

**Step 1: Zmień komendę Pyright w workflow**

Zamień:

```yaml
- name: Pyright
  run: .venv/bin/pyright config
```

na:

```yaml
- name: Pyright
  run: .venv/bin/pyright main.py config
```

**Step 2: Zweryfikuj lokalnie przed push**

Run:

```bash
.venv/bin/pyright main.py config
```

Expected:
- zielono lokalnie,
- brak nowych błędów.

**Step 3: Commit**

```bash
git add .github/workflows/quality-gates.yml
git commit -m "ci: expand pyright gate to main entrypoint"
```

---

### Task 3: Dopnij dokumentację quality gate do realnego stanu

**Objective:** Usunąć sprzeczność między sekcjami `docs/README-DEV.md`.

**Files:**
- Modify: `docs/README-DEV.md`

**Step 1: Ujednolić opis gate**

Zostaw jedną prawdę spójną z CI:
- obecnie gate = `pyright config`,
- po wykonaniu Task 2 gate = `pyright main.py config`.

W trakcie aktualizacji nie zostawiaj w dalszej części pliku starych zdań typu:
- `pyright must pass with zero warnings`
- ogólników sugerujących strict gate dla całego repo, jeśli CI tego nie egzekwuje.

**Step 2: Urealnij komendy lokalne**

Wstaw komendę parity zgodną z aktualnym etapem planu:

**przed Task 2:**

```bash
.venv/bin/pyright config
```

**po Task 2:**

```bash
.venv/bin/pyright main.py config
```

**Step 3: Commit**

```bash
git add docs/README-DEV.md
git commit -m "docs: align pyright gate guidance with actual CI scope"
```

---

### Task 4: Wypchnij i potwierdź GH run po rozszerzeniu gate

**Objective:** Zweryfikować, że rozszerzony gate nadal kończy się szybko i przewidywalnie.

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
- brak regresji do szerokiego scope.

---

### Task 5: Zapisz roadmapę dalszego rozszerzania strict coverage

**Objective:** Po odblokowaniu `main.py` zachować kontrolowany plan zdejmowania długu typów.

**Files:**
- Create: `docs/plans/pyright-strict-expansion-roadmap.md`

**Recommended phases:**
1. `config`
2. `main.py`
3. `models`
4. małe moduły w `core`
5. reszta `core`
6. `gui`
7. `tests`

**Suggested success metric per phase:**
- lokalny Pyright dla danego scope = zielony,
- workflow scope rozszerzony dopiero po lokalnym potwierdzeniu,
- GH run zielony po każdym etapie,
- dokumentacja zaktualizowana przy każdej zmianie scope.

**Suggested minimal file skeleton:**

```md
# Pyright Strict Expansion Roadmap

## Current enforced CI scope
- pyright config

## Next candidate scope
- pyright main.py config

## Phase backlog
1. models
2. core (small modules first)
3. core (remaining)
4. gui
5. tests

## Rules
- never widen CI scope before local green run
- no blanket suppressions
- no whole-repo strict claim until CI actually enforces it
```

---

## Verification checklist

- [x] `pyrightconfig.json` nie ma już `include: ["."]`
- [x] `.github/workflows/quality-gates.yml` używa zawężonego scope Pyrighta
- [x] `.venv/bin/pyright config` jest zielone lokalnie
- [x] GH run kończy Pyright szybko i przewidywalnie
- [ ] `.venv/bin/pyright main.py config` jest zielone lokalnie
- [ ] workflow rozszerzono do `main.py config`
- [ ] `docs/README-DEV.md` jest całkowicie spójny z faktycznym gate’em
- [ ] istnieje zapisany plan dalszego rozszerzania strict-checka

---

## Notes / guardrails

- Nie wracaj do `include: ["."]`.
- Nie wyłączaj Pyrighta z CI.
- Nie rozszerzaj workflow na kolejny obszar, dopóki lokalny scope nie jest zielony.
- Nie zostawiaj w dokumentacji deklaracji szerszych niż realny CI.
- Najbliższy sensowny krok to **naprawa `main.py`**, nie ruszanie od razu `core`, `gui`, `models` ani `tests`.
