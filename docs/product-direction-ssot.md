# PromptManager — Product Direction SSOT

Status: superseded
Updated: 2026-04-25
Canonical file: `docs/product-ssot.md`

## Cel
Ten dokument jest krótkim SSOT kierunku produktu PromptManager.
Ma trzymać jedną jasną odpowiedź na pytania:
- czym ten produkt jest,
- dla kogo jest,
- jaki jest jego główny workflow,
- co jest rdzeniem,
- co jest warstwą zaawansowaną,
- czego świadomie teraz nie pozycjonujemy jako core.

---

## Jednozdaniowa definicja produktu
**PromptManager to local-first system do przechwytywania, porządkowania, wyszukiwania i ponownego używania prompt assets, z operacyjną warstwą uruchomień i routingu rozwijaną jako wsparcie tego rdzenia.**

---

## Główny użytkownik
PromptManager jest przede wszystkim dla:
- power usera AI,
- prompt engineera,
- developera lub operatora, który pracuje z wieloma promptami,
- osoby, która chce mieć trwały, przeszukiwalny katalog promptów zamiast trzymać je w notatkach, chat history i przypadkowych plikach.

To nie jest dziś produkt pozycjonowany przede wszystkim jako:
- ogólny chatbot desktopowy,
- pełny agent platform,
- ogólny AI workbench do wszystkiego.

---

## Główny workflow produktu
Rdzeń produktu to pętla:
1. **capture** — zapisz prompt lub draft,
2. **organize** — nadaj mu tytuł, kategorię, tagi, kontekst i metadane,
3. **retrieve** — znajdź go później przez search, recent lub semantic retrieval,
4. **inspect** — oceń jego treść, historię, podobieństwo i przydatność,
5. **reuse** — skopiuj, otwórz w workspace albo użyj jako bazę do kolejnej pracy.

To jest podstawowa pętla, której produkt nie może rozmywać.

---

## Rdzeń produktu (primary)
Za core PromptManagera uznajemy:
- quick capture,
- draft → reusable asset promotion,
- katalog promptów,
- metadane i klasyfikację,
- semantic retrieval,
- recent/search flows,
- inspect/details view,
- copy/open/reuse workflows,
- wersjonowanie i provenance prompt assets,
- local-first persistence i przewidywalną własność danych użytkownika.

Jeżeli nowa funkcja nie wzmacnia tego obszaru, nie jest funkcją core.

---

## Warstwa zaawansowana (advanced ops)
PromptManager może zawierać i rozwijać funkcje zaawansowane, ale są one drugą warstwą, nie główną definicją produktu.

Do tej warstwy należą:
- prompt execution,
- refinement flows,
- scenario generation,
- benchmarki i porównania,
- analytics,
- sharing,
- workspace helpers,
- historia wykonań,
- model routing i diagnostics,
- bardziej rozbudowane operacje na prompt chains.

Te funkcje są wartościowe, ale powinny być prezentowane jako:
- advanced,
- optional,
- wspierające główną pętlę katalogu i reuse.
- rozwijane dopiero wtedy, gdy nie osłabiają czytelności warstwy core.

---

## Kierunek strategiczny
Docelowy kierunek produktu to:

**catalog-first, ops-second, automation-third**

To znaczy:
- PromptManager ma być najpierw najlepszym domem dla prompt assets,
- potem czytelną warstwą operacyjną do ich testowania, uruchamiania i porównywania,
- a dopiero później kontrolowaną warstwą automatyzacji opartą na tym samym modelu produktu.

Jeżeli pojawia się konflikt priorytetów między:
- kolejną funkcją workbench/ops,
- a poprawą capture/retrieval/reuse,

preferencję ma warstwa core katalogu.

---

## Czego świadomie nie pozycjonujemy teraz jako core
Na ten moment PromptManager nie powinien być opisywany jako:
- „pełny AI operating system”,
- „uniwersalny AI workbench”,
- „agent platform”,
- „desktop app do wszystkiego z LLM-ami”.

Takie capability mogą częściowo istnieć w kodzie, ale nie są obecnie główną obietnicą produktu.

---

## Zasady rozwoju produktu
Przy ocenie nowych funkcji zadawaj pytania:
1. Czy to wzmacnia capture?
2. Czy to wzmacnia retrieval?
3. Czy to wzmacnia inspect?
4. Czy to wzmacnia reuse?
5. Czy to poprawia jakość życia prompt asset jako trwałego obiektu?

Jeśli odpowiedź brzmi głównie „nie”, funkcja należy raczej do advanced layer albo later.

---

## Priorytety krótkoterminowe
Najwyższy produktowy ROI mają:
- lepszy front door do capture,
- lepsza czytelność draft/promote flows,
- mocniejsze sygnały podobieństwa i duplikatów,
- wygodniejsze search/recent/reuse,
- lepszy inspect view dla prompt assetów,
- deterministyczne settings, routing i embedding resolution,
- czytelna warstwa diagnostics i effective config,
- uporządkowanie informacji architektonicznej w GUI między core a advanced.

---

## Priorytety średnioterminowe
Po dopracowaniu rdzenia warto rozwijać:
- execution jako warstwę wspierającą reuse,
- benchmarki i scenariusze jako warstwę walidacji promptów,
- analytics jako warstwę informacji zwrotnej o użyciu i skuteczności,
- historia wykonań i porównania runów,
- CLI/API automation surfaces bez rozbijania local-first charakteru,
- sharing i collaboration surfaces bez rozbijania local-first charakteru.

---

## Reguła komunikacyjna dla README i roadmapy
README, roadmapa i komunikacja produktu powinny najpierw sprzedawać:
- prompt assets,
- retrieval,
- inspect,
- reuse,
- local-first ownership.

Funkcje execution/benchmark/analytics/routing/diagnostics powinny być opisywane jako rozszerzenia lub warstwa zaawansowana.
Automation surfaces powinny być komunikowane dopiero jako kolejna warstwa, a nie nowa tożsamość produktu.

---

## Decyzja operacyjna
Jeśli pojawi się wątpliwość „czy PromptManager to katalog promptów czy workbench?”, obowiązuje odpowiedź:

**najpierw katalog prompt assets, dopiero potem workbench capabilities.**
