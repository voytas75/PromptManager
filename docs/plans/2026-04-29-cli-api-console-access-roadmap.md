# PromptManager CLI / API Console Access Roadmap

Status: proposed

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Otworzyć kolejny bounded execution cycle w Stage 4 tak, aby AI asystent i operator mogli korzystać z PromptManagera z konsoli w zakresie najważniejszych operacji na danych prompt assetów i ich ocenie, bez budowania drugiego produktu obok GUI i bez omijania istniejącego modelu ustawień/diagnostyki.

**Architecture:** Ten roadmap zostaje w warstwie controlled automation surfaces i preferuje cienkie CLI/API wrappers nad już istniejącymi seamami w `core/`, `cli/parser.py`, `cli/commands.py`, `main.py` oraz ewentualnie małych read-only helperach w managerze. Celem nie jest nowe RPC, serwer agentowy ani background daemon, tylko spokojne domknięcie brakujących konsolowych entry points do danych, które są realnie potrzebne asystentowi: listowanie, odczyt, bounded retrieval, inspect/reuse-ready export oraz lekkie operacje na run evidence.

**Tech Stack:** Python 3.13, argparse CLI, PromptManager core manager/repository helpers, pytest, Ruff, Pyright, existing SSOT docs under `docs/`.

---

## Why this roadmap exists

Live repo/SSOT review confirms:
- `docs/product-roadmap-ssot.md` utrzymuje **Stage 4 — Controlled automation surfaces** jako naturalny kolejny etap po domknięciu ostatniego cyklu entry-clarity,
- `docs/product-ssot.md` mówi wprost o `stronger CLI workflows`, `exportable or scriptable run operations`, oraz `API/internal service boundaries where justified`,
- repo ma już kilka ważnych headless seamów (`prompt-add`, `catalog-import`, `suggest`, `history-analytics`, `benchmark`, `refresh-scenarios`, `prompt-chain-*`),
- obecna warstwa CLI dobrze obsługuje import, benchmarky i analytics snapshoty, ale nadal nie daje jeszcze spójnego, prostego zestawu read-first entry points do katalogu promptów dla AI asystenta pracującego z konsoli,
- poprzedni etap skupił się na GUI/core asset loop clarity; teraz sensowny ruch to nie kolejna mikro-poprawka GUI, tylko bounded console-access leverage nad tym samym modelem danych.

Short version:

**assets first -> trustworthy operations -> controlled console access for prompt data**

---

## Product intent for this cycle

Ten cykl ma odpowiedzieć na pytanie:

> skoro PromptManager jest local-first canonical home dla prompt assetów, to jaki minimalny zestaw CLI/API entry points trzeba jeszcze domknąć, żeby AI asystent albo operator mógł z konsoli sensownie odczytać, znaleźć, obejrzeć i przygotować prompt asset do reuse/refine bez otwierania GUI?

Nowy cykl wzmacnia trzy rzeczy:

1. **console-readable asset access** — podstawowe dane promptów muszą być dostępne w spokojnych, deterministycznych formach z CLI,
2. **assistant-usable retrieval/inspect surface** — asystent ma móc znaleźć prompt i odczytać jego kluczowe pola bez reverse-engineeringu SQLite lub GUI-only widget seams,
3. **single product model** — CLI/API ma reuse’ować istniejące `core` i shared analytics seams zamiast wprowadzać nowy równoległy backend.

---

## Constraints

Do not add:
- osobnego serwera HTTP tylko po to, żeby „mieć API”,
- background daemon lub hidden sync layer,
- drugiego modelu danych tylko dla CLI/API,
- GUI-bypass semantics, które omijają istniejące settings/diagnostics,
- szerokiego agent frameworka w repo PromptManager,
- write-heavy automation bez jasnego bounded operator path.

Prefer first:
- `cli/parser.py`
- `cli/commands.py`
- `main.py`
- istniejące helpery w `core/`
- shared analytics / catalog seams already used by GUI and CLI

---

## Confirmed baseline before this cycle

Assume already delivered and do not reopen without a failing regression:
- `prompt-add` i `catalog-import` already cover several add/import paths,
- `history-analytics` already exposes shared evaluation cues (`decision`, `next`, `freshness`) where they travel through shared analytics,
- `suggest` already exists as a bounded semantic retrieval entry point,
- benchmark / prompt-chain / scenario-refresh console seams already exist,
- CLI trust surfaces (`--print-settings`) already follow the trust-first posture.

This cycle should build on those seams instead of replacing them.

---

## Proposed direction: CLI first, API only where justified

**Recommended default:** najpierw zrobić brakujące **CLI-first read/retrieve/inspect surfaces**.

Why:
- to najniższy koszt i najmniejsze ryzyko względem SSOT,
- AI asystent pracujący lokalnie i tak najłatwiej skorzysta z deterministic CLI output,
- repo ma już sprawdzony parser/commands/test seam dla bounded console slices,
- można szybko zweryfikować realną użyteczność zanim powstanie presja na formalne API.

**API/internal service boundary** dopiero jako kolejny krok, jeśli po CLI-first cycle wyjdzie realna potrzeba:
- współdzielonego structured output,
- stabilniejszego machine-to-machine contract niż plain console text,
- lub lekkiego in-process helper layer, który upraszcza kilka CLI komend naraz.

---

## Candidate operator jobs for console access

Najbardziej prawdopodobne console-critical jobs dla AI asystenta:
- list prompts with lightweight filters,
- show one prompt in full reusable form,
- export one prompt as structured JSON/text for downstream reuse,
- run bounded search/suggest and inspect why a result matters,
- inspect recent/useful run evidence tied to one prompt,
- compare enough metadata to decide reuse vs refine before opening GUI.

These jobs suggest read-heavy work first, not mutation-heavy automation.

---

## Roadmap stages for this cycle

### Stage A — Library read surfaces

Cel: dać prosty i deterministyczny konsolowy dostęp do listowania i odczytu prompt assetów.

Obszary:
- prompt list / summary command,
- prompt show / inspect command,
- optional output modes (`text` first, maybe `json` when justified),
- bounded filtering by existing fields only.

Constraint:
- reuse existing manager/repository seams,
- no new ranking/storage model.

### Stage B — Assistant-usable retrieval surfaces

Cel: zrobić retrieval lepiej używalnym z konsoli dla asystenta niż samo obecne `suggest`.

Obszary:
- clarify when to use text search vs semantic suggest,
- maybe one thin `prompt-find` or similar command over existing search/retrieval helpers,
- readable result summaries with prompt identity + enough inspect cues.

Constraint:
- no retrieval redesign,
- no separate recommendation engine.

### Stage C — Inspect / export / reuse-ready readout

Cel: dać bounded headless surface do pełnego odczytu jednego prompta i przygotowania go do reuse/refine przez asystenta.

Obszary:
- one prompt detail/export seam,
- machine-usable output where justified,
- preserve shared truth for fields like title/body/description/tags/source/context.

Constraint:
- keep semantics identical to the canonical prompt record,
- no second serialization dialect unless clearly needed.

### Stage D — Run evidence console helpers

Cel: domknąć najważniejsze read-only operacje na run evidence dla pojedynczego prompta.

Obszary:
- one bounded prompt-focused history/evidence view over existing history seams,
- maybe compact per-prompt latest-run or recent-run list,
- preserve existing `history-analytics` as broader snapshot surface.

Constraint:
- no new analytics engine,
- no broad dashboard duplication in CLI.

### Stage E — Internal API boundary probe

Cel: ocenić, czy po CLI-first slices potrzebna jest mała internal API/service boundary dla structured assistant access.

Obszary:
- identify repeated formatting/lookup logic,
- maybe extract one read-only shared helper contract used by multiple CLI commands,
- explicitly decide whether external API is still unnecessary.

Constraint:
- only if repetition or machine-usage justifies it,
- prefer in-process shared helper over network service.

---

## Recommended first slice

### Slice 1 — Prompt show/read CLI seam v1

**Status:** done (expanded 2026-04-29)

**Delivered:**
- added `prompt-show <prompt_id-or-name>` as a deterministic read-only CLI surface,
- lookup now supports UUID first and exact-name fallback,
- output includes: `id`, `name`, `description`, `category`, `tags`, `source`, `active`, and `context` when present,
- missing prompt returns exit code `4`.

**Notes:**
- This lands the CLI-first single-prompt read seam without introducing JSON mode or new internal API helpers.
- Exact-name fallback is now included in the same bounded slice because it remained thin and reused the existing repository list seam.

### Slice 2 — Prompt find/list CLI seam v1

**Status:** done (2026-04-29)

**Delivered:**
- added `prompt-find <query>` as a deterministic read-only CLI surface,
- case-insensitive matching currently scans `name`, `description`, `category`, and `tags`,
- added `--limit` with default `10`,
- compact output format is `id | name | [category] | tags`.

**Notes:**
- This slice stays CLI-first and reuses `repository.list()` without adding a new core search seam.
- It is intentionally lightweight and complements `prompt-show` rather than replacing semantic `suggest`.

### Slice 3 — Prompt show JSON output v1

**Status:** done (2026-04-29)

**Delivered:**
- added `prompt-show --json` for structured single-prompt reads,
- JSON payload reuses `prompt.to_record()` to preserve canonical prompt fields,
- output is pretty-printed and deterministic for console/agent consumption.

**Notes:**
- This slice stays CLI-first and does not introduce a new internal API helper.
- It extends the existing `prompt-show` seam rather than creating a separate export command.

**Why this first:**
- to najbliższa brakująca funkcja względem „konsolowego dostępu do danych”,
- import/add już istnieje, ale odczyt jednego prompta nadal nie jest tak prosty jak powinien,
- daje najwyższy leverage dla asystenta bez projektowania nowego API,
- tworzy naturalny anchor pod późniejsze `list/find/export/history` slices.

**Good shape:**
- one new command like `prompt-show` or `prompt-get`,
- lookup by prompt UUID and optionally exact name as fallback,
- compact text output first: id, title, description, category, tags, source, context/body, active flag, maybe lightweight lineage/run pointers if already cheap,
- deterministic failure when prompt is not found.

**Likely files:**
- Modify: `cli/parser.py`
- Modify: `cli/commands.py`
- Maybe modify: `main.py` (only if compatibility re-export seam is needed)
- Modify: `tests/test_main_entry.py`
- Maybe modify: `README.md`
- Modify: `docs/CHANGELOG.md`

**Verification target:**
- targeted CLI tests in `tests/test_main_entry.py`,
- live `python -m main --help`,
- full repo verification bundle when the slice lands.

---

## Success criteria for the whole roadmap

This cycle is successful if, by the end:
- an AI assistant can access core prompt data from CLI without scraping SQLite directly,
- the new console surfaces clearly reuse the existing PromptManager product model,
- read/retrieve/inspect tasks from headless workflows become simpler than opening the GUI for routine access,
- no second product model or opaque automation layer was introduced.

---

## Decision note

Based on current SSOT + repo state, **CLI-first is the recommended next roadmap**, not external API-first.

API/internal shared boundary should be reconsidered only after at least one or two concrete CLI read/access slices show what structured machine access is still missing.
