# PromptManager — Next Slice Brief

Date: 2026-04-05
Status: delivered
Slice name: Usage Cue v1
Primary source: `docs/product-boundary-ssot.md`
Supporting sources:
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/CHANGELOG.md`
- `README.md`

## Recommended slice

Add one bounded **Usage Cue** improvement to the existing **prompt detail / inspect** flow.

When an operator opens a prompt in the existing detail view, PromptManager should expose one compact **When to use** cue when existing prompt data already contains a short credible usage signal.

This slice should improve inspect clarity without widening into AI-generated guidance, a new persistence model, or a dedicated guidance surface.

## Why this now

The recent alignment work already strengthened the prompt asset core loop materially:
- Quick Capture
- Recent Reopen
- draft inspection cues
- Draft Promote / Normalize v1
- Promote-time Similar Prompt Check v1
- Quick Reuse Handoff v1
- Capture Provenance v1

That leaves a smaller but still meaningful inspect gap:

> after opening a prompt, can the operator tell quickly whether this is the right prompt to use now?

This slice strengthens **Inspect + Reuse confidence** by surfacing one bounded usage signal directly in the main detail flow.

## User problem solved

Without a usage cue, the operator may still need to:
- read the whole prompt body,
- infer intent from the title,
- or inspect deeper metadata/scenario surfaces
before deciding whether the prompt is the right one.

The goal is simple:

> open prompt → quickly understand when to use it

## Boundaries

### Do now
Implement exactly one narrow detail-view improvement:

1. Add one compact **When to use** cue to the existing prompt detail view.
2. Derive the cue only from existing prompt fields/data.
3. Prefer already-saved usage-like signals such as:
   - scenario text
   - description
   - example input
4. Show the cue only when the signal is short and credible.
5. If no credible signal exists, show nothing.
6. Keep the cue bounded inside the existing detail flow.
7. Add focused regression coverage for the bounded usage-cue happy paths.

### Strong defaults for v1
- Prefer existing fields over any new `when_to_use` schema.
- Prefer one compact line over a new panel or dialog.
- Prefer no cue over low-quality or invented filler text.
- Prefer bounded heuristics over AI summarization.
- Keep inspect behavior stable outside the added cue.

## Do later
- stronger best-fit heuristics
- generated usage cues
- scenario summarization
- usage cues in search/list views
- richer guidance authoring surfaces
- confidence-ranked multiple usage cues

## Acceptance checks

1. Existing detail view shows a compact **When to use** cue when sufficient existing signal is present.
2. The cue is derived only from existing prompt data.
3. Prompts without credible signal do not show a noisy or fabricated cue.
4. No schema migration or new persistence model is introduced.
5. Rendering remains bounded within the current detail flow.
6. Focused regression coverage protects:
   - cue visible when signal exists
   - cue hidden when signal is absent
   - bounded rendering inside the current detail flow

## Rollback

Rollback should be one isolated patch:
- remove the usage cue rendering from the detail view,
- remove the minimal helper logic,
- remove the focused regression tests,
- leave the rest of the inspect flow untouched.

## Anti-goals

- Do not add AI-generated prompt coaching.
- Do not add a new `when_to_use` persistence field.
- Do not redesign the whole detail view.
- Do not widen into search/list/workspace rollout.
- Do not add a recommendation engine.
- Do not touch Quick Capture, Promote, chains, analytics, sharing, web search, TTS, or unrelated execution behavior.

## Suggested implementation posture

Keep the slice small and honest:
- one compact cue,
- existing data only,
- bounded heuristics,
- focused tests,
- no peripheral expansion.

The product win is not smarter behavior.
The product win is letting the operator recognize the right prompt faster.

## Definition of done

The slice is done when:
- an operator can open a prompt and quickly see when it should be used,
- the cue relies only on existing data,
- no fabricated cue appears when signal is weak,
- the implementation stays bounded,
- the focused tests pass,
- nothing peripheral expands.

## Implementation note

Delivered on 2026-04-05 as a compact **When to use** cue in the existing prompt detail flow.

Implementation derives the cue only from existing prompt data, preferring short saved scenarios and falling back to other already-stored fields when the signal remains compact and credible. Rendering stays bounded inside the shared detail surface, and focused regression coverage protects visible, absent, and bounded-rendering cases.
