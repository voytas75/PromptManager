# PromptManager — Next Slice Brief

Date: 2026-04-04
Status: proposed
Slice name: Quick Reuse Handoff v1
Primary source: `docs/product-boundary-ssot.md`
Supporting sources:
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/CHANGELOG.md`
- `README.md`

## Recommended slice

Add one bounded **Quick Reuse Handoff** improvement to the existing **prompt detail** flow.

When an operator opens a prompt in the existing detail view, PromptManager should expose two immediate reuse actions:
- **Copy Prompt Body**
- **Open in Workspace**

This slice should stay strictly within the current detail flow and should not widen into execution redesign, export expansion, or analytics.

## Why this now

The 2026-04-04 alignment work already strengthened the earlier parts of the core loop:
- Quick Capture
- Recent Reopen
- draft inspection cues
- Draft Promote / Normalize v1
- Promote-time Similar Prompt Check v1
- README alignment with the prompt-asset core loop

That means the largest remaining bounded gap in the core loop is no longer capture.
It is **quick reuse**.

PromptManager is already closer to the SSOT, but it still does not make the final transition fully boring:

> find prompt → inspect prompt → reuse prompt immediately

This slice strengthens **Reuse** without turning execution into the product center.

## User problem solved

Today, a user can find and inspect a good prompt asset, but the handoff into actual reuse is still weaker than it should be.

The user often still has to:
- open the prompt,
- decide they want to reuse it,
- move manually to another surface,
- or rely on context-menu actions that are not part of the primary detail experience.

This creates unnecessary friction at the exact moment the product should feel most direct.

The goal is simple:

> open prompt → reuse immediately in one obvious move

## Boundaries

### Do now
Implement exactly one narrow improvement in the existing detail flow:

1. Add one compact reuse action row to the existing prompt detail view.
2. Expose exactly two actions:
   - **Copy Prompt Body**
   - **Open in Workspace**
3. Reuse existing prompt payload semantics where possible:
   - primary payload should remain the prompt body / existing main text field
   - fallback behavior should follow the current application’s established copy/open semantics rather than inventing a new rule
4. Reuse existing clipboard and workspace handoff paths where possible.
5. Keep `Open in Workspace` as a **non-executing** handoff.
6. Add focused regression coverage for the bounded happy paths.

### Strong defaults for v1
- Prefer the existing detail surface over a new panel, dialog, or tab.
- Prefer two obvious actions over a richer reuse menu.
- Prefer existing wiring and helper methods over new abstractions.
- Prefer `Open in Workspace` without auto-run.
- Prefer no migration, no persistence change, and no new analytics.

## Do later
- additional copy/export modes
- explicit last-used / reuse cues
- reuse actions in list/search views
- richer execution handoff options
- deeper workspace simplification or reuse ergonomics

## Acceptance checks

1. The selected prompt detail view shows both:
   - `Copy Prompt Body`
   - `Open in Workspace`
2. `Copy Prompt Body` copies the expected prompt payload.
3. `Open in Workspace` seeds the workspace input with the expected prompt payload.
4. `Open in Workspace` does **not** auto-run the prompt.
5. The slice works for:
   - standard prompts
   - prompts created through quick-capture / draft-promote flows
6. The slice adds no new persistence, migrations, or schema changes.
7. Focused regression coverage protects the bounded reuse happy path.

## Rollback

Rollback should be one isolated patch:
- remove the detail-view reuse action row,
- remove the minimal callback wiring,
- remove the focused regression tests,
- leave capture, inspect, promote, and workspace execution flows otherwise unchanged.

## Anti-goals

- Do not add `Run Prompt` to the detail view in this slice.
- Do not redesign the detail layout broadly.
- Do not refactor the workspace architecture beyond the smallest required handoff seam.
- Do not add reuse analytics or telemetry expansion.
- Do not widen into export/share systems.
- Do not touch chains, sharing, web search, or TTS.
- Do not turn this into a general “reuse framework”.

## Suggested implementation posture

Keep the slice small and honest:
- one existing detail surface,
- two direct actions,
- one non-executing workspace handoff,
- focused tests,
- no peripheral expansion.

The product win is not more power.
The product win is making the final step of the core loop feel immediate.

## Definition of done

The slice is done when:
- a user can open a prompt and immediately copy it or seed it into the workspace,
- the workspace handoff does not auto-run,
- the implementation stays bounded,
- the focused tests pass,
- nothing peripheral expands.
