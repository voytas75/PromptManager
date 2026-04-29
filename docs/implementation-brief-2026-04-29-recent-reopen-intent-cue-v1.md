# Implementation Brief — Recent Reopen Intent Cue v1

## Goal
Sprawdzić, czy dialog `Recent Prompts` potrzebuje jednego małego, widocznego reopen-intent cue ponad istniejące `Modified ... • Category: ...`, tak aby operator szybciej rozumiał, że wybiera prompt do ponownego otwarcia i dalszej pracy, bez rozbudowy w history browser.

## Why this slice now
Po domknięciu `Quick Capture entry clarity v1` najbliższym seamem w tym samym cycle jest `Recent Prompts`. Runtime ma już sensowny summary line i visible row metadata, ale nie jest jeszcze potwierdzone testem, czy obecna forma jest wystarczająco czytelna operacyjnie, czy potrzebny jest jeden dodatkowy cue na istniejącym dialog seamie.

## Scope in
- `gui/dialogs/recent_prompts.py`
- `tests/test_recent_prompts.py`
- jeden mały visible cue albo guard-only closure

## Scope out
- brak history browsera
- brak nowych sort/filter controls
- brak persistence changes
- brak zmian w reopen semantics

## Acceptance checks
1. Najpierw pojawia się RED probe na jednym konkretnym reopen-intent expectation.
2. Jeśli runtime już jest wystarczająco czytelny, slice zamyka się guard-only bez fake churn.
3. Jeśli nie, zmiana pozostaje lokalna do dialog seam.
4. Targeted `tests/test_recent_prompts.py` przechodzi.
5. Ruff + `py_compile` dla touched files przechodzą.
