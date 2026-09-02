# PromptManager — Next Slice Brief

Date: 2026-04-11
Status: delivered
Slice name: Search Match Highlight v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/analysis-review-2026-04-11-quick-capture-real-input-review-v1.md`

## Recommended slice

Add one bounded **Search Match Highlight** improvement to the existing **main prompt list**.

When an operator types a normal text search and browses the current prompt-list results, PromptManager should subtly highlight matching query terms inside the already visible row text:
- prompt title
- existing preview line, when present

This slice should improve **retrieve → inspect confidence** without changing ranking, adding badges, or widening into a list redesign.

## Why this now

The stronger recent bounded slices have already improved:
- Quick Capture cleanup for the obvious wrapper-noise cases
- detail-view inspection cues
- quick reuse handoff and reuse tooltip clarity
- template-aware workspace handoff for templated prompts

At the same time:
- the latest Quick Capture analysis explicitly says **no new cleanup slice is justified yet**
- README / positioning cleanup has already been treated as delivered baseline
- the product review still points toward **retrieval / inspection ergonomics** as the best next bounded area

That leaves one practical retrieval gap:

> after search narrows the list, can the operator see *why* a result matches without opening several similar prompts one by one?

This slice strengthens retrieval clarity directly at the point of selection.

## User problem solved

Today, the operator can search and can see a bounded preview line.
But when several prompts look similar, the list still makes the operator do extra mental work to spot the relevant result.

The goal is simple:

> search → scan results faster → open the right prompt sooner

## Boundaries

### Do now
Implement exactly one narrow retrieval-surface improvement:

1. Add subtle match highlighting to the existing main prompt-list rows.
2. Highlight only text that already appears in the row:
   - title
   - existing preview line
3. Trigger highlighting only when an explicit text search is active.
4. Keep the match logic deterministic and boring:
   - plain text tokens or normalized substring matches
   - case-insensitive
5. Keep selection, sorting, ranking, and filtering unchanged.
6. Keep the current list layout intact.
7. Add focused regression coverage for active-search and no-search states.

### Strong defaults for v1
- Prefer one muted emphasis style over colorful or multi-state highlighting.
- Prefer the existing prompt list delegate over a new list surface.
- Prefer no highlight over fuzzy or misleading emphasis.
- Prefer simple text matching over semantic or token-score explanations.
- Prefer no extra metadata chips, icons, or labels.

## Do later
- multi-color match categories
- ranking explanations
- semantic-match reasons in the list
- badges for source, draft, or usage state
- cross-surface highlighting in dialogs beyond the main prompt list
- broader search UX redesign

## Acceptance checks

1. When a normal prompt search is active, matching query text is subtly highlighted in the main prompt-list title.
2. When a preview line is present and contains the match, the preview can also highlight the matching text.
3. When no search is active, the list renders as it does today.
4. Sorting, filtering, selection, and ranking behavior remain unchanged.
5. No new persistence, schema, or search-index fields are introduced.
6. The change stays inside the current prompt-list search/rendering seam.
7. Focused regression coverage protects:
   - highlight visible during active search
   - no highlight when search is empty
   - bounded rendering with preview-line coexistence

## Rollback

Rollback should be one isolated patch:
- remove the match-highlight rendering from the prompt list delegate/model seam
- remove any minimal query-to-delegate wiring required for the highlight
- remove the focused regression tests
- leave retrieval preview, selection, filtering, and detail view untouched

## Anti-goals

- do not redesign the prompt list
- do not change search ranking or similarity logic
- do not add AI-generated match explanations
- do not add badges, icons, or extra metadata rows
- do not widen into execution, analytics, chains, sharing, or voice
- do not reopen Quick Capture cleanup in the same pass

## Suggested implementation posture

Keep the slice small and honest:
- one existing list surface
- one subtle visual cue
- existing search text only
- focused tests
- no adjacent cleanup wave

The product win is not prettier rendering.
The product win is helping the operator recognize the right prompt faster from the retrieval surface already in use.

## Implementation note

Delivered on 2026-04-12 as a bounded active-search emphasis pass in the existing main prompt list.

Implementation stays inside the prompt-list search/rendering seam:
- active plain-text search exposes bounded title/preview match spans from the model
- the delegate renders one subtle emphasis treatment for matching visible text
- sorting, filtering, selection, ranking, and list layout remain unchanged
- focused regression coverage protects active-search and no-search behavior
