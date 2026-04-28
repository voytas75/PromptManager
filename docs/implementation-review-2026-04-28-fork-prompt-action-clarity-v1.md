# Implementation Review — fork-prompt-action-clarity-v1

## Delivered
- dodano tooltip dla akcji `Fork Prompt` w menu kontekstowym promptu
- tooltip wyjaśnia, że akcja tworzy fork zachowujący lineage i od razu otwiera go do edycji
- bez zmian w logice forka, callbackach, storage i flow edytora

## Runtime change
- `gui/prompt_actions_controller.py`
  - `fork_action.setToolTip("Create a fork linked to this prompt and open it for editing.")`

## Tests
- `tests/test_prompt_actions_controller.py`
  - dodano test `test_show_context_menu_explains_fork_prompt_action`
  - RED potwierdzone: tooltip był pusty
  - GREEN potwierdzone po zmianie runtime

## Verification
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_prompt_actions_controller.py::test_show_context_menu_explains_fork_prompt_action -q`
  - `1 passed`

## Scope control
- zmiana została utrzymana lokalnie w seamie menu kontekstowego
- nie zmieniano `PromptEditorFlow.fork_prompt()`
- nie zmieniano statusów, enable/disable, storage ani lineage modelu
