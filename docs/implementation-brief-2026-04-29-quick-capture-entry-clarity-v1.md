# Implementation Brief — Quick Capture Entry Clarity v1

## Goal
Dodać jeden mały, zawsze widoczny cue w dialogu `Quick Capture`, który wyjaśnia operatorowi, że to entry point dla surowego promptu / query i że PromptManager robi tylko bounded cleanup obvious wrappers przed zapisaniem draftu.

## Why this slice now
`Quick Capture` jest najbliższym entry seamem do asset-first core loop. Runtime ma już realne cleanup semantics, ale komunikuje je głównie przez placeholder pola body. To zostawia małą, ale czytelną lukę w operator posture: użytkownik nie widzi od razu, że może wkleić raw input i że aplikacja nie próbuje uruchamiać pełnego transcript parsera.

## Scope in
- `gui/dialogs/quick_capture.py`
- `tests/test_quick_capture_dialog.py`
- mały, user-visible helper text / cue w samym dialogu
- bez zmiany cleanup semantics

## Scope out
- brak nowych przycisków, opcji lub wizard steps
- brak zmian w persistence/modelu draftu
- brak zmian w unwrap/strip heuristics
- brak CLI/headless parity

## Acceptance checks
1. Dialog pokazuje jeden kompaktowy cue poza samym placeholderem body input.
2. Cue komunikuje raw-input posture i bounded cleanup obvious wrappers.
3. Cleanup behavior pozostaje bez zmian.
4. Focused offscreen pytest przechodzi razem z całym `tests/test_quick_capture_dialog.py`.
5. Ruff + `py_compile` dla dotkniętych plików przechodzą.
