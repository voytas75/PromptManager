# PromptManager — Implementation Brief

Date: 2026-04-28
Status: ready-for-implementation
Feature: Recent Row Metadata Visibility v1
Primary source: bounded follow-up to `docs/implementation-brief-2026-04-04-recent-reopen.md`

## Goal

Implement one bounded **Recent Row Metadata Visibility v1** improvement so the existing `Recent Prompts` dialog shows one visible secondary metadata line per row, helping the operator choose the right recent prompt without relying on tooltip hover.

## Product intent

This slice strengthens the existing core loop at:
- retrieve,
- inspect,
- reuse.

It should reduce friction between:
- “I know this was touched recently”
- and
- “I can tell which recent prompt I want before reopening it.”

## Scope

### In scope
- one visible secondary metadata line inside each recent-prompt row
- reuse of existing prompt metadata only
- compact operator-facing wording based on current `last_modified` and `category`
- focused regression coverage for row text rendering

### Out of scope
- new persistence for recent metadata
- usage scoring
- recency reasons like `recently edited` / `recently captured`
- favorites / pinning
- history browser expansion
- backend ranking or selection changes
- status-toasts after reopen

## Recommended UX posture

Keep the dialog compact.
Do not redesign the flow.

Suggested v1 row shape:
- first line: prompt title
- second line: `Modified <timestamp> • Category: <name>`

The timestamp should stay aligned with the dialog’s existing compact UTC formatting.

## Likely implementation seam

### UI
- `gui/dialogs/recent_prompts.py`
  - switch from plain string list rows to widget-backed rows
  - keep current ordering and selection behavior unchanged
  - reuse `_format_timestamp()`

### Tests
- `tests/test_recent_prompts.py`
  - add one focused rendering test for visible row metadata
  - keep the existing ordering/selection handoff test unchanged

## Acceptance checks

1. `Recent Prompts` still shows the same ordered prompts and selection flow.
2. Each visible row exposes one compact metadata line without hover.
3. The metadata line uses only existing prompt fields.
4. The timestamp stays in compact UTC format.
5. Focused tests protect the rendering seam.

## Rollback

Rollback should be one isolated patch:
- remove the custom row widget rendering
- restore title-only recent rows
- remove the focused rendering regression test
- keep the original recent reopen flow intact

## Notes for implementation

- Prefer the smallest Qt seam that makes the metadata visible.
- Keep tooltip support if it remains effectively free.
- Do not widen into richer filtering, provenance, or action menus.
