# Implementation Brief — duplicate-prompt-action-clarity-v1

## Goal
Dowieźć mały usability slice dla istniejącej akcji `Duplicate Prompt`, żeby operator od razu rozumiał, że akcja tworzy zwykłą kopię promptu, a nie fork z lineage.

## Why this slice now
Po doprecyzowaniu `Fork Prompt` i `Similar Prompts` naturalnym kolejnym krokiem jest odróżnienie zwykłego duplikowania od forka. To poprawia legibility menu kontekstowego bez zmiany workflow.

## Scope
- lokalna zmiana w `gui/prompt_actions_controller.py`
- dodać tooltip dla akcji `Duplicate Prompt`
- wording ma jasno mówić, że akcja tworzy editable copy bez semantyki forka
- bez zmian w callbacku, editor flow i storage

## Out of scope
- zmiana logiki `duplicate_callback`
- zmiana flow zapisu lub nazewnictwa kopii
- nowy dialog, lineage UI, dodatkowe statusy

## Expected files
- `gui/prompt_actions_controller.py`
- `tests/test_prompt_actions_controller.py`
- `docs/CHANGELOG.md`

## Acceptance
- RED: nowy test sprawdza tooltip dla `Duplicate Prompt`
- GREEN: runtime ustawia tooltip
- targeted pytest green
- nearby smoke green
