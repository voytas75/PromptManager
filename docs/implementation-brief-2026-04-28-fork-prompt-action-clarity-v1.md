# Implementation Brief — fork-prompt-action-clarity-v1

## Goal
Dowieźć mały usability slice dla istniejącej akcji `Fork Prompt`, poprawiający legibility tego, co operator dostaje po kliknięciu.

## Why this slice now
`Fork Prompt` jest blisko core reuse/refine flow, ale obecnie nie komunikuje wprost skutku akcji. W odróżnieniu od `Duplicate Prompt`, fork ma semantykę lineage i dalszej edycji, więc krótki tooltip pomaga odróżnić go od zwykłego kopiowania.

## Scope
- lokalna zmiana w istniejącym seamie menu kontekstowego w `gui/prompt_actions_controller.py`
- dodać tooltip dla akcji `Fork Prompt`
- wording ma wyjaśniać, że akcja tworzy fork z lineage i otwiera go do edycji
- bez zmian w callbackach, storage, managerze i flow zapisu

## Out of scope
- zmiana enable/disable logiki `Fork Prompt`
- nowy dialog, nowy status, nowe lineage UI
- zmiany w `PromptEditorFlow.fork_prompt()`

## Expected files
- `gui/prompt_actions_controller.py`
- `tests/test_prompt_actions_controller.py`
- `docs/CHANGELOG.md`

## Acceptance
- RED: nowy test sprawdza tooltip dla `Fork Prompt`
- GREEN: tooltip jest ustawiany w runtime
- targeted pytest green
- git status i diff czytelne
