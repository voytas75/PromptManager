# CLI Settings Diagnostics Reorder v1

## Goal

Tighten the `--print-settings` trust surface so the diagnostics block appears before low-level config details, making the effective runtime state visible first.

## Why this slice

PromptManager already exposes compact diagnostics with source and precedence labels, but the current CLI summary still leads with raw config sections before the operator sees `Overall status`, blocking items, and next steps.

This slice stays inside **Priority 2 — Trust infrastructure** because it improves:
- effective config visibility
- diagnostics clarity
- reduction of ambiguous runtime noise

## Scope in

- reorder the CLI settings summary so diagnostics comes earlier
- keep all existing diagnostics content, labels, and masking rules
- update focused tests to lock the new output order
- update docs only where the user-visible CLI validation flow changes

## Scope out

- no new diagnostics fields
- no GUI changes
- no settings precedence model changes
- no persistence or schema changes

## Files expected

- Modify: `cli/settings_summary.py`
- Modify: `tests/test_settings_summary.py`
- Modify: `tests/test_main_entry.py`
- Update: `README-DEV.md`
- Update: `docs/CHANGELOG.md`
- Update: `docs/plans/2026-04-25-roadmap-implementation-plan.md`

## Acceptance checks

1. `python -m main --no-gui --print-settings` shows the diagnostics block before detailed LiteLLM / embeddings / integrations sections.
2. `Overall status` and any `Next steps` remain unchanged in wording.
3. Existing masking behaviour for secrets remains unchanged.
4. Focused CLI summary tests assert the reordered trust-first layout.
5. No new warnings, lint errors, or type errors are introduced.
