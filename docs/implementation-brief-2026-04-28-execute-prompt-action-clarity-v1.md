# Implementation Brief — execute-prompt-action-clarity-v1

## Goal
Dowieźć mały usability slice dla istniejącej akcji `Execute Prompt`, żeby operator od razu rozumiał, że akcja uruchamia prompt od razu z jego bieżącym payloadem.

## Why this slice now
Po doprecyzowaniu `Duplicate Prompt`, `Fork Prompt`, `Similar Prompts` i `Execute as Context…` naturalnym kolejnym krokiem jest wyjaśnienie bezpośredniego wykonania. To domyka czytelność najważniejszych akcji w menu kontekstowym.

## Scope
- lokalna zmiana w `gui/prompt_actions_controller.py`
- dodać tooltip dla akcji `Execute Prompt`
- wording ma jasno mówić, że akcja uruchamia prompt od razu przy użyciu jego stored body / fallback payload
- bez zmian w execution flow, callbackach i storage

## Out of scope
- zmiana logiki `execute_prompt_from_body`
- zmiana enable/disable logiki `Execute Prompt`
- nowe statusy, nowe dialogi, zmiany execution controller

## Expected files
- `gui/prompt_actions_controller.py`
- `tests/test_prompt_actions_controller.py`
- `docs/CHANGELOG.md`

## Acceptance
- RED: nowy test sprawdza tooltip dla `Execute Prompt`
- GREEN: runtime ustawia tooltip
- targeted pytest green
- nearby smoke green
