# PromptManager — Session Restart Brief

Date: 2026-04-06
Status: active restart brief
Purpose: po resecie sesji używać tego pliku jako krótkiego, bieżącego kontekstu dla dalszych prac nad bounded slice'ami.

## Jak używać tego pliku po resecie

Jeśli Wojtek powoła się na ten plik, traktuj go jako:
- bieżący restart point,
- skrót aktualnego stanu slice'ów,
- zestaw wytycznych do dalszej pracy,
- dokument nadrzędny dla tej serii małych wdrożeń, chyba że nowszy brief go zastąpi.

## Product center / SSOT posture

PromptManager ma pozostać:

> local-first canonical home for prompt assets

Priorytetem są tylko elementy wzmacniające core loop:
- capture
- normalize
- retrieve
- inspect
- reuse
- refine

Nie rozlewać scope'u w stronę:
- agent platform
- AI workstation drift
- chains jako centrum produktu
- analytics/dashboard expansion jako główny tor
- broad sharing / voice / novelty UX bez bezpośredniego wpływu na core loop

## Source anchors

Przy kolejnych decyzjach i briefach najpierw czytać:
- `docs/product-boundary-ssot.md`
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/product-backlog-ssot.md`
- ten plik

Jeśli potrzebny jest kontekst wdrożeniowy dla reuse:
- `docs/implementation-review-2026-04-06-reuse-polish-v1.md`
- `docs/CHANGELOG.md`
- `README.md`

## Co już jest dowiezione i nie powinno być planowane od nowa

### Core-loop slices already delivered
1. `Quick Capture to Draft`
2. `Recent Reopen`
3. `Draft Promote / Normalize v1`
4. `Reuse Polish v1`
5. `Copy Prompt terminology consistency v1`
6. `Copy Prompt docs cleanup v1`
7. `Capture Provenance v1`
8. `Usage Cue v1`
9. `Retrieval Preview v1`
10. `Similar Match Preview v1`
11. `Context Lead Usage Cue v1`
12. `Reuse Payload Tooltip v1`
13. `Credible Source Cue v1`

### Practical meaning
- draft capture działa,
- recent reopen działa,
- promote draft działa,
- detail-view quick reuse działa,
- capture provenance działa,
- usage cue działa,
- retrieval preview działa,
- similar-match preview działa,
- context-lead usage cue działa,
- reuse payload tooltip działa,
- credible-source cue działa,
- body-only copy label jest spójny jako `Copy Prompt`,
- aktywne docs są już wyrównane do `Copy Prompt`.

## Reuse terminology rule

Dla body-only prompt copy obowiązuje jedna nazwa:

**`Copy Prompt`**

Stare nazwy typu:
- `Copy Prompt Body`
- `Copy Prompt Text`

traktować jako legacy / history only, nie jako aktualny wording dla nowych briefów, review i docs.

## Working style for next slices

Każdy kolejny slice ma być:
- mały,
- bounded,
- boring,
- lokalny w skutkach,
- bez bocznych refaktorów,
- z jasnym DoD,
- z rollbackiem,
- z anti-goals.

Preferowana kolejność pracy:
1. wybrać dokładnie jeden bounded next slice,
2. zawsze zrobić krótki build brief przed implementacją,
3. jeśli delegacja jest użyta, zrobić brief delegacyjny dla Codexa,
4. wdrożyć,
5. zrobić krótki implementation review,
6. zawsze zaktualizować stosowne docs dla dowiezionego slice'a, minimalnie i bez docs-cleanup driftu.

## Hard constraints for next implementation passes

- nie re-implementować slice'ów już dowiezionych,
- nie zaczynać implementacji bez briefu dla danego slice'a,
- nie kończyć slice'a bez aktualizacji stosownych dokumentów,
- nie robić broad cleanup przy okazji małego celu,
- nie mieszać kilku zmian produktowych w jednym slice'ie,
- nie zmieniać semantyki istniejących flow bez osobnego briefu,
- nie przebudowywać editor/workspace/retrieval bez wyraźnej decyzji,
- nie używać docs cleanup jako pretekstu do scope expansion.

## Validation posture

Dla wdrożeń kodowych preferować:
- focused tests dla slice'a,
- a jeśli zmiana dotyka live seams lub UI labels szerzej, pełniejszą walidację repo jeśli koszt jest rozsądny.

W odpowiedzi końcowej zawsze podać:
- co zmieniono,
- listę plików,
- wyniki walidacji,
- ocenę, czy slice pozostał bounded.

## Next-session default posture

Po resecie nie wracać do starych tematów z rozpędu.
Najpierw:
1. potwierdzić, że ten plik jest restart briefem,
2. wskazać jeden sensowny następny bounded slice,
3. dopiero potem iść w build brief / delegację / implementację.

## Recommended operator wording after reset

Jeśli Wojtek napisze w stylu:
- "jedziemy dalej wg tego briefu"
- "użyj restart briefa z 2026-04-06"
- "kontynuuj PromptManager z pliku restartowego"

traktuj ten plik jako obowiązujący kontekst roboczy dla następnego małego slice'a.

## Intent of this file

Ten dokument nie jest roadmapą.
To jest restart point + guardrails dla następnego etapu małych wdrożeń.
