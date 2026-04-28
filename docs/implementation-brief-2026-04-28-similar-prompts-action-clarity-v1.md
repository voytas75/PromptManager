# Implementation Brief — similar-prompts-action-clarity-v1

## Goal
Dowieźć mały usability slice dla istniejącej akcji `Similar Prompts`, żeby operator od razu rozumiał, że uruchamia ścieżkę rekomendacji podobnych promptów, a nie zwykłe wyszukiwanie tekstowe.

## Why this slice now
`Similar Prompts` leży blisko core retrieval/reuse path PromptManagera. Akcja jest ważna semantycznie, ale w samym menu kontekstowym nie tłumaczy swojego efektu. Mały tooltip poprawia legibility bez zmiany workflow.

## Scope
- lokalna zmiana w `gui/prompt_actions_controller.py`
- dodać tooltip dla akcji `Similar Prompts`
- wording ma jasno mówić o recommendation results dla podobnych promptów
- bez zmian w callbacku, presenterze, rankingach i search flow

## Out of scope
- zmiana logiki `similar_callback`
- zmiana statusów presenter/listy
- nowy dialog, nowy panel, nowe ranking heuristics

## Expected files
- `gui/prompt_actions_controller.py`
- `tests/test_prompt_actions_controller.py`
- `docs/CHANGELOG.md`

## Acceptance
- RED: nowy test sprawdza tooltip dla `Similar Prompts`
- GREEN: runtime ustawia tooltip
- targeted pytest green
- nearby smoke green
