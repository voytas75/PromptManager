# Implementation Brief — Promote Draft advisory entry wording v1

## Goal
Dodać jeden mały, zawsze widoczny cue w `Promote Draft`, który spokojnie wyjaśnia advisory posture tego entry pointu: najpierw review existing match, albo continue as new.

## Why this slice now
Stage C aktywnego roadmapu dotyczy entry clarity dla `Promote Draft`. Dialog ma już similarity/advisory semantics i rozróżnienie `Promote as New` vs `Open Existing Match`, ale summary zaczyna od wyniku similarity, nie od krótkiego operator-facing meaning cue dla samego entry pointu.

## In scope
- jeden compact wording refinement na istniejącym summary seamie w `gui/dialogs/draft_promote.py`
- jeden focused RED test w `tests/test_draft_promote_dialog.py`
- changelog + roadmap update po green

## Out of scope
- zmiany similarity engine
- zmiany button semantics
- compare/duplicate workflow
- nowe controls, metadata, persistence albo CLI parity

## Expected files
- `gui/dialogs/draft_promote.py`
- `tests/test_draft_promote_dialog.py`
- `docs/CHANGELOG.md`
- `docs/plans/2026-04-29-capture-reopen-refine-entry-clarity-roadmap.md`

## Acceptance checks
- RED: nowy targeted test failuje na brakującym advisory-entry wording
- GREEN: targeted test przechodzi po minimalnej zmianie copy
- smoke: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_draft_promote_dialog.py -q`
- quality: `ruff check` + `python -m py_compile` dla dotkniętych plików
