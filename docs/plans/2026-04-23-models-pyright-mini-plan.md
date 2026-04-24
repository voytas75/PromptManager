# Models Pyright Expansion Mini Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Przygotować bezpieczne rozszerzenie typed coverage z `pyright main.py config` do `pyright main.py config models` bez rozwalania zielonego CI.

**Architecture:** Zakres `models` ma obecnie 87 błędów Pyright i nie nadaje się do jednego dużego fixa. Najbezpieczniej iść od najmniejszych, najbardziej izolowanych plików do najcięższych modeli opartych o payloady `dict[str, Any]`. Celem tego mini-planu nie jest od razu pełne wdrożenie, tylko ustawienie kolejności prac i pierwszego bezpiecznego cięcia.

**Tech Stack:** Pyright 1.1.408, Python 3.13, repo PromptManager.

---

## Current verified state

- aktualny enforced scope CI: `pyright main.py config`
- roadmapa bazowa: `docs/plans/pyright-strict-expansion-roadmap.md`
- lokalny probe dla następnego scope:

```bash
.venv/bin/pyright main.py config models
```

- wynik potwierdzony 2026-04-23: **87 errors**
- największe hotspoty:
  - `models/prompt_model.py`
  - `models/prompt_chain_model.py`
  - `models/response_style.py`
  - `models/category_model.py`

## Recommended file order

1. `models/category_model.py`
2. `models/response_style.py`
3. `models/prompt_chain_model.py`
4. `models/prompt_model.py`

## Why this order

- `category_model.py` ma mały, lokalny problem typu `list[Unknown]` i nadaje się na pierwszy zielony slice.
- `response_style.py` jest średnie, ale ma dość czytelny zestaw problemów: prywatne helpery + kolekcje bez typu.
- `prompt_chain_model.py` już dotyka bardziej złożonych payloadów kroków i konwersji z mappingów.
- `prompt_model.py` jest największym polem minowym; powinno wejść dopiero po ustaleniu wzorca typowania dla wcześniejszych plików.

---

### Task 1: Domknij `models/category_model.py`

**Objective:** Usunąć najprostszy błąd `list[Unknown]`, żeby ustawić wzorzec pracy na models.

**Files:**
- Modify: `models/category_model.py`
- Test: jeśli istnieją odpowiednie testy modelu, uruchom tylko je; jeśli nie, wystarczy lokalny Pyright dla pliku

**Observed issue:**
- `default_tags` jest inferowane jako `list[Unknown]`

**Step 1: Uruchom Pyright tylko dla pliku**

```bash
.venv/bin/pyright models/category_model.py
```

**Expected:**
- pojedynczy lub bardzo mały zestaw błędów związanych z `default_tags`

**Step 2: Doprecyzuj jawny typ kolekcji**

Najpewniej potrzebne będzie coś w rodzaju:

```python
default_tags: list[str] = []
```

albo jawne typowanie pomocniczej zmiennej, jeśli lista jest budowana dynamicznie.

**Step 3: Zweryfikuj**

```bash
.venv/bin/pyright models/category_model.py
```

**Expected:**
- `0 errors`

**Step 4: Commit**

```bash
git add models/category_model.py
git commit -m "fix: type category model for pyright"
```

---

### Task 2: Domknij `models/response_style.py`

**Objective:** Usunąć prywatne użycia helperów i kolekcje `Unknown` w średnio małym pliku.

**Files:**
- Modify: `models/response_style.py`
- Verify if needed: plik z helperami importowanymi przez `response_style.py`

**Observed issues:**
- `reportPrivateUsage` dla helperów importowanych z innego modułu
- `tags`, `examples`, `metadata_dict` mają częściowo nieznane typy

**Step 1: Uruchom Pyright tylko dla pliku**

```bash
.venv/bin/pyright models/response_style.py
```

**Step 2: Napraw prywatne helpery bez hacków**

Preferowana kolejność:
1. jeśli helper ma być publiczny między modelami, zmień nazwę/eksport na publiczny w module źródłowym,
2. jeśli helper nie powinien być współdzielony, przenieś lokalną logikę do `response_style.py`,
3. nie używaj szerokiego ignorowania `reportPrivateUsage`.

**Step 3: Jawnie otypuj kolekcje i mappingi**

Typowe bezpieczne kierunki:

```python
tags: list[str] = []
examples: list[str] = []
metadata_dict: dict[str, Any]
```

Dopasuj do realnego kontraktu pliku.

**Step 4: Zweryfikuj**

```bash
.venv/bin/pyright models/response_style.py
```

**Step 5: Commit**

```bash
git add models/response_style.py
git commit -m "fix: type response style model for pyright"
```

---

### Task 3: Domknij `models/prompt_chain_model.py`

**Objective:** Uporządkować typed parsing dla kroków chaina i zmapować kontrakt payloadów.

**Files:**
- Modify: `models/prompt_chain_model.py`
- Test: odpowiednie testy chain model, jeśli istnieją

**Observed issues:**
- `MutableMapping[Unknown, Unknown]`
- `steps` / `steps_payload` jako `list[Unknown]`
- `get()` na payloadach bez jawnego typu mappingu

**Step 1: Uruchom Pyright tylko dla pliku**

```bash
.venv/bin/pyright models/prompt_chain_model.py
```

**Step 2: Zdefiniuj minimalny kontrakt wejścia**

Preferuj jeden z dwóch wariantów:

```python
from typing import Any

StepPayload = Mapping[str, Any]
```

albo mały `TypedDict`, jeśli pola są stabilne i krótkie.

**Step 3: Otypuj iterację po krokach**

Zamiast pozwalać Pyrightowi zgadywać:

```python
steps_payload: list[StepPayload]
for step_payload in steps_payload:
    ...
```

**Step 4: Zweryfikuj**

```bash
.venv/bin/pyright models/prompt_chain_model.py
```

**Step 5: Commit**

```bash
git add models/prompt_chain_model.py
git commit -m "fix: type prompt chain model for pyright"
```

---

### Task 4: Domknij `models/prompt_model.py`

**Objective:** Uporządkować największy hotspot typów w `models` po ustaleniu wzorca na wcześniejszych plikach.

**Files:**
- Modify: `models/prompt_model.py`
- Verify: powiązane helpery modeli, jeśli potrzebne
- Test: odpowiednie testy prompt model / serialization, jeśli istnieją

**Observed issues:**
- dużo `get()` na `Any` / `dict[Unknown, Unknown]`
- wiele kolekcji inferowanych jako `list[Unknown]` / `dict[Unknown, Unknown]`
- pojedyncze `reportUnnecessaryIsInstance`

**Step 1: Nie próbuj naprawiać całości naraz**

Podziel plik logicznie na 3 mini-fale:
1. kolekcje i lokalne zmienne (`tags`, `scenarios`, `recent_prompts`, `settings`)
2. metadata/payload mapping (`metadata`, `base`, `get(...)`)
3. cleanup zbędnych `isinstance`

**Step 2: Najpierw ustaw jawne typy pośrednie**

Przykładowy kierunek:

```python
metadata: Mapping[str, Any] | None
base: dict[str, Any]
```

Zamiast zostawiać Pyrightowi inferencję z niesprecyzowanego JSON-like payloadu.

**Step 3: Zweryfikuj po każdej mini-fali**

```bash
.venv/bin/pyright models/prompt_model.py
```

Nie przechodź dalej, dopóki bieżąca fala nie jest zielona lub wyraźnie mniejsza.

**Step 4: Commit**

```bash
git add models/prompt_model.py
git commit -m "fix: type prompt model for pyright"
```

---

### Task 5: Zbiorcza weryfikacja i decyzja o rozszerzeniu gate

**Objective:** Potwierdzić, że `models` jest gotowe do wejścia do lokalnego enforced scope.

**Files:**
- Verify: `models/`
- Modify later if green: `.github/workflows/quality-gates.yml`
- Modify later if green: `docs/README-DEV.md`
- Modify later if green: `docs/plans/pyright-strict-expansion-roadmap.md`

**Step 1: Uruchom pełny next-scope probe**

```bash
.venv/bin/pyright main.py config models
```

**Success condition:**

```text
0 errors, 0 warnings, 0 informations
```

**Step 2: Jeśli zielone, dopiero wtedy rozszerz workflow**

Docelowa komenda CI:

```yaml
- name: Pyright
  run: .venv/bin/pyright main.py config models
```

**Step 3: Urealnij dokumentację**

Zaktualizuj:
- `docs/README-DEV.md`
- `docs/plans/pyright-strict-expansion-roadmap.md`

**Step 4: Push i GH verification**

```bash
git push origin master
gh run list --repo voytas75/PromptManager --limit 5
gh run view --repo voytas75/PromptManager <run_id>
```

---

## Guardrails

- nie rozszerzaj CI do `models`, dopóki lokalne `pyright main.py config models` nie jest zielone
- nie używaj szerokich `# type: ignore`
- nie zaczynaj od `prompt_model.py`
- nie mieszaj tej pracy z GUI analytics albo kolejnymi zmianami settings precedence
- po każdym pliku utrzymuj repo w stanie zielonym dla lokalnych checków, które już są domknięte

## Recommended first slice

**Pierwszy bezpieczny krok:** `models/category_model.py`

Powód:
- najmniejszy known hotspot,
- dobry kandydat na szybki wzorzec typowania,
- najmniejsze ryzyko rozwalenia zachowania runtime.
