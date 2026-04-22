# Pyright Gate Recovery Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Przywrócić zielony quality gate na GitHubie bez udawania pełnej zgodności całego repo z `pyright strict`, a jednocześnie zostawić jasną ścieżkę do etapowego porządkowania długu typów.

**Architecture:** Rozdzielamy problem na dwa poziomy. Najpierw odblokowujemy CI przez ograniczenie zakresu Pyrighta w workflow do obszaru o najwyższym ROI i najniższym długu typów. Potem osobno utrzymujemy roadmapę dochodzenia do szerszego strict-checka dla `core`, `gui`, `models` i `tests`.

**Tech Stack:** GitHub Actions, Pyright 1.1.408, Python 3.13, repo PromptManager.

---

## Current diagnosis

- `pyrightconfig.json` z `include: ["."]` powodował zbyt szeroki skan; to zostało już naprawione.
- Ostatni GH run `24802329186` kończy się szybko (~1m34s), więc problem performance został zbity.
- Pyright nadal failuje na **1634 błędach**, głównie w:
  - `core/`
  - `gui/`
  - `models/`
  - `tests/`
- Najczęstsze klasy błędów:
  - `reportUnknownVariableType`
  - `reportUnknownArgumentType`
  - `reportUnknownMemberType`
  - `reportMissingParameterType`
  - `reportPrivateUsage`
  - `reportUnnecessaryIsInstance`
- Developer guide nadal deklaruje pełny strict gate, więc po zmianie workflow trzeba zaktualizować też dokumentację SSOT dla jakości.

---

### Task 1: Ogranicz GH gate do minimalnego, stabilnego scope

**Objective:** Zmienić workflow tak, by GitHub Actions uruchamiał Pyright tylko dla najwęższego sensownego zakresu, który można utrzymać zielony teraz.

**Files:**
- Modify: `.github/workflows/quality-gates.yml`
- Test: lokalne odpalenie tej samej komendy Pyright

**Step 1: Zmień komendę Pyright w workflow**

Zamień:

```yaml
      - name: Pyright
        run: .venv/bin/pyright
```

na:

```yaml
      - name: Pyright
        run: .venv/bin/pyright main.py config
```

**Dlaczego ten zakres:**
- `config` już lokalnie przechodził bez błędów,
- `main.py` ma małą powierzchnię i szybki feedback,
- to daje realny typowy gate zamiast fikcji lub totalnego czerwonego stanu.

**Step 2: Zweryfikuj lokalnie**

Run:

```bash
.venv/bin/pyright main.py config
```

Expected:
- jeśli są błędy w `main.py`, zobaczysz mały, konkretny zestaw do poprawy,
- nie powinno być już tysięcy błędów.

**Step 3: Commit**

```bash
git add .github/workflows/quality-gates.yml
git commit -m "ci: narrow pyright gate to stable entrypoints"
```

---

### Task 2: Napraw mały zestaw błędów z `main.py`

**Objective:** Doprowadzić `pyright main.py config` do stanu zielonego.

**Files:**
- Modify: `main.py`
- Verify: `config/` (jeśli Pyright wskaże dodatkowe drobne problemy)

**Known likely errors from local reproduction:**
- unused imports:
  - `PromptManagerSettings`
  - `SettingsError`
- unknown / missing parameter types przy helperach w `main.py`
- partially unknown `run_default_mode`

**Step 1: Usuń nieużywane importy**

Usuń importy, których Pyright zgłasza jako nieużywane.

**Step 2: Dodaj jawne anotacje helperom w `main.py`**

Jeżeli występuje np. `_setup_logging(config_path)`, doprecyzuj parametr jako `Path | str` albo właściwy typ zgodny z użyciem.

Przykład wzorca:

```python
from pathlib import Path


def _setup_logging(config_path: str | Path) -> None:
    ...
```

**Step 3: Doprecyzuj import / typ `run_default_mode`**

Jeśli Pyright widzi `Unknown`, użyj jednego z podejść:
- popraw typ w miejscu definicji funkcji,
- albo zaimportuj ją z modułu, który ma pełne anotacje,
- albo dodaj lokalny Protocol / Callable typ tylko jeśli to konieczne.

**Step 4: Uruchom lokalnie**

Run:

```bash
.venv/bin/pyright main.py config
```

Expected:
- `0 errors, 0 warnings, 0 informations`

**Step 5: Commit**

```bash
git add main.py config
git commit -m "fix: satisfy narrowed pyright gate"
```

---

### Task 3: Zsynchronizuj dokumentację quality gate

**Objective:** Urealnić dokumentację tak, żeby repo nie deklarowało czegoś, czego CI faktycznie nie egzekwuje.

**Files:**
- Modify: `docs/README-DEV.md`

**Step 1: Zmień opis quality gate**

Zastąp sformułowania typu:
- „Pyright must pass with zero warnings”
- „strict mode for the whole repo”

na komunikat zgodny z rzeczywistością, np.:

```md
- **Type Checking**: GitHub quality gate currently enforces `pyright main.py config` as the stable typed entrypoint set. Wider strict coverage for `core`, `gui`, `models`, and `tests` is tracked as incremental technical-debt reduction work.
```

**Step 2: Dodaj lokalną komendę parity**

W sekcji toolchain dopisz realny odpowiednik CI, np.:

```bash
.venv/bin/pyright main.py config
```

**Step 3: Commit**

```bash
git add docs/README-DEV.md
git commit -m "docs: align pyright guidance with enforced gate"
```

---

### Task 4: Wypchnij i zweryfikuj GH run

**Objective:** Potwierdzić, że nowy workflow działa i Pyright nie blokuje już pipeline’u przez dług albo zbyt szeroki scope.

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
- `Pyright` kończy się szybko,
- jeśli `main.py config` jest zielone, pipeline przechodzi do `Pytest`.

**Step 3: Jeśli fail nadal dotyczy tylko `main.py config`, popraw od razu i zrób follow-up commit**

---

### Task 5: Zapisz roadmapę rozszerzania strict coverage

**Objective:** Nie gubić długu typów po odblokowaniu gate’a.

**Files:**
- Create or update: `docs/plans/pyright-strict-expansion-roadmap.md` (lub osobna notka)

**Recommended phases:**
1. `main.py` + `config`
2. `models`
3. `core/execution.py` i małe moduły w `core`
4. reszta `core/prompt_manager`
5. `gui`
6. `tests`

**Suggested success metric per phase:**
- brak nowych błędów Pyright w danym obszarze,
- scope w workflow rozszerzony o kolejny katalog,
- zielony GH po każdym etapie.

---

## Verification checklist

- [ ] `pyrightconfig.json` nie ma już `include: ["."]`
- [ ] `.github/workflows/quality-gates.yml` używa zawężonego scope Pyrighta
- [ ] `.venv/bin/pyright main.py config` jest zielone lokalnie
- [ ] GH run kończy Pyright szybko i przewidywalnie
- [ ] `docs/README-DEV.md` odpowiada faktycznemu gate’owi
- [ ] jest zapisany plan dochodzenia do szerszego strict-checka

---

## Notes / guardrails

- Nie wracaj do `include: ["."]`.
- Nie próbuj „naprawiać” 1634 błędów jednym commitem.
- Nie wyłączaj całkiem Pyrighta z CI — stracisz typed smoke test dla entrypointów.
- Jeżeli pojawi się presja na szybkie „zielone”, najbezpieczniejszy kompromis to **mały typed gate + jawna roadmapa rozszerzania**.
