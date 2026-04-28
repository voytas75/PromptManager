# PromptManager Retrieval/Discovery Roadmap Prep

> **Purpose:** przygotować docs i SSOT pod nowy roadmap execution cycle bez otwierania jeszcze samego roadmapu implementacyjnego.

**Status:** active prep
**Updated:** 2026-04-27
**Related canonical file:** `docs/product-ssot.md`

---

## What is already settled

Potwierdzone po audycie repo:
- `docs/product-ssot.md` pozostaje kanonicznym SSOT i nadal utrzymuje asset-first posture,
- dotychczasowy execution ledger `docs/plans/2026-04-26-asset-to-run-to-refine-coherence-v3-roadmap.md` jest praktycznie domknięty,
- aktualny product direction nie wymaga zmiany warstw ani redefinicji produktu,
- najbliższy sensowny nowy focus produktowy to **retrieval/discovery leverage**, a nie dalsze mikro-polish na inspect/workspace handoff.

---

## Why the next roadmap should move here

Nowy roadmap powinien wzmacniać to, co SSOT już traktuje jako core:
- `Retrieve` — odnaleźć prompt później po tekście, metadanych, recent use albo semantic similarity,
- `Retrieval and discovery` — pełnotekstowe wyszukiwanie, metadata filtering, semantic retrieval, related/similar prompt discovery,
- `reuse becomes faster than rediscovery` jako praktyczny efekt produktu.

Po ostatnim cyklu inspect/run/refine operator dostaje już lepsze bounded decision cues.
Najbardziej naturalna następna luka produktowa to nie kolejna warstwa guidance, tylko **czytelniejszy i bardziej godny zaufania powrót do właściwego prompt assetu z poziomu listy / search / retrieval flow**.

---

## Scope for the next roadmap

Nowy roadmap powinien preferować:
- search-result legibility,
- retrieval/discovery trust cues,
- lepsze uzasadnienie dlaczego wynik jest pokazany,
- szybsze przejście z retrieval do inspect/reuse,
- bounded similarity / related-result clarity bez nowego workflow.

Nowy roadmap nie powinien startować od:
- nowych dashboardów,
- ciężkiej analityki searchowej,
- osobnego retrieval cockpit,
- ukrytej zmiany rankingu bez legible explanation,
- nowego persistence model tylko dla explanation cues.

---

## SSOT alignment made before opening the roadmap

Przed otwarciem nowego roadmapu doprecyzowano:
- `docs/product-ssot.md` Priority 1 tak, aby mocniej akcentował `retrieval/discovery ergonomics`, zwłaszcza `search-result legibility` i szybszy reuse,
- `README.md` Current focus tak, aby jawnie sygnalizował retrieval/discovery clarity jako aktualny focus obok core prompt loop i trust surfaces.

To jest doprecyzowanie priorytetu wykonawczego, nie zmiana definicji produktu.

---

## Recommended shape of the first roadmap slice

Najlepszy pierwszy slice dla nowego roadmapu:
- **search match reason clarity**

Good shape:
- reuse existing prompt-list / preview / inspect seams,
- nie zmieniać rankingu w pierwszym kroku,
- dodać tylko jeden bounded operator-facing cue wyjaśniający, dlaczego dany wynik jest trafny,
- sprawdzić parity między listą wyników i inspect entry path tylko tam, gdzie semantic jest już shared.

Przykładowe bounded semantics do rozważenia:
- `Matched in title`
- `Matched in source`
- `Matched in scenario`
- `Semantic match`

Wybór konkretnego wording i seam pozostaje do potwierdzenia podczas właściwego roadmap draftu po recon runtime/tests.

---

## Definition of ready for the next roadmap

Nowy roadmap można otworzyć, gdy:
- SSOT nadal jasno stawia prompt assets i retrieval w centrum,
- README nie sugeruje sprzecznego focusu,
- poprzedni v3 roadmap nie jest już traktowany jako aktywny source of work,
- pierwszy retrieval/discovery slice da się sformułować jako bounded improvement, a nie redesign.
