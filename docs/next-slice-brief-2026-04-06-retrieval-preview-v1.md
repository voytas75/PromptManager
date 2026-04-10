# PromptManager — Next Slice Brief

Date: 2026-04-06
Status: delivered
Slice name: Retrieval Preview v1
Primary source: `docs/product-boundary-ssot.md`
Supporting sources:
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/CHANGELOG.md`
- `README.md`

## Recommended slice

Add one bounded **Retrieval Preview** improvement to the existing **main prompt list**.

When an operator browses or searches the prompt catalog, PromptManager should expose one compact secondary preview line for a prompt when existing prompt data already contains a short credible distinguishing signal.

This slice should improve **retrieve → inspect confidence** before opening the detail view, without widening into a list redesign, new schema, or AI-generated summaries.

## Why this now

The recent bounded slices already strengthened the core loop materially:
- Quick Capture
- Recent Reopen
- draft inspection cues
- Draft Promote / Normalize v1
- Promote-time Similar Prompt Check v1
- Quick Reuse Handoff / Reuse Polish v1
- Capture Provenance v1
- Usage Cue v1
- Title Quality v1
- Copy Prompt wording/docs consistency

That leaves one practical gap in the core loop:

> can the operator recognize the right prompt quickly from the retrieval surface itself, before opening several similar items one by one?

Right now the main prompt list is still relatively thin as a retrieval surface. If titles are similar, the operator may still need to click into multiple prompts just to figure out which one is the right asset.

This slice strengthens **Retrieve** first and **Inspect** second by adding one bounded distinguishing hint directly where selection happens.

## User problem solved

Without a compact retrieval preview, the operator may still need to:
- open multiple similar prompts,
- rely too heavily on title/category alone,
- or bounce between list and detail views just to identify the right asset.

The goal is simple:

> search or browse → recognize the likely right prompt faster

## Boundaries

### Do now
Implement exactly one narrow retrieval-surface improvement:

1. Add one compact secondary preview line to the existing main prompt list rows.
2. Derive the preview only from existing prompt data.
3. Prefer already-saved distinguishing signals such as:
   - description
   - scenario text
   - source
4. Show the preview only when the signal is short and credible.
5. If no credible signal exists, keep the row clean rather than inventing filler.
6. Keep the change bounded to the shared prompt list surface already used for prompt browsing/selection.
7. Add focused regression coverage for the bounded preview happy paths.

### Strong defaults for v1
- Prefer existing fields over any new retrieval-preview schema.
- Prefer one muted secondary line over badges, chips, or cards.
- Prefer no preview over noisy or fabricated text.
- Prefer deterministic truncation/flattening over AI summarization.
- Keep sort/filter/selection behavior unchanged.

## Do later
- search-term highlighting
- richer badges for source/draft/usage state
- multi-line or card-like list redesign
- confidence-ranked preview selection heuristics
- preview text in additional surfaces beyond the shared prompt list
- generated preview summaries

## Acceptance checks

1. The existing prompt list can show one compact secondary preview line when sufficient existing signal is present.
2. The preview is derived only from existing prompt data.
3. Prompts without credible signal do not show noisy or fabricated preview text.
4. No schema migration or new persistence model is introduced.
5. Sorting, filtering, and selection behavior remain unchanged.
6. Focused regression coverage protects:
   - preview visible when signal exists
   - preview hidden when signal is absent
   - bounded rendering/truncation in the existing list surface

## Rollback

Rollback should be one isolated patch:
- remove the secondary preview rendering from the prompt list,
- remove the minimal helper/delegate logic,
- remove the focused regression tests,
- leave capture, promote, inspect, and reuse flows untouched.

## Anti-goals

- Do not redesign the whole prompt list.
- Do not add AI-generated retrieval summaries.
- Do not add a new preview persistence field.
- Do not widen into analytics, prompt parts, sharing, chains, or workspace behavior.
- Do not change search ranking, similarity logic, or sort semantics in this slice.
- Do not add badges, icons, or multiple metadata rows just because they are nearby.

## Suggested implementation posture

Keep the slice small and honest:
- one secondary preview line,
- existing data only,
- bounded formatting,
- focused tests,
- no adjacent cleanup wave.

The product win is not a prettier list.
The product win is helping the operator identify the right prompt faster from the retrieval surface itself.

## Definition of done

The slice is done when:
- an operator can distinguish similar prompts more easily from the main list,
- the preview relies only on existing data,
- no fabricated preview appears when signal is weak,
- the implementation stays bounded,
- the focused tests pass,
- nothing peripheral expands.

## Implementation note

Delivered as a bounded secondary preview line in the shared main prompt list.

Implementation derives one compact preview only from existing prompt data, preferring description, then scenario text, then credible source, with deterministic flattening/truncation and focused regression coverage for visible, absent, and bounded rendering paths.
