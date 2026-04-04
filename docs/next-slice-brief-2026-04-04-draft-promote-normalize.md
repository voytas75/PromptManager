# PromptManager — Next Slice Brief

Date: 2026-04-04
Status: proposed
Slice name: Draft Promote / Normalize v1
Primary source: `docs/product-boundary-ssot.md`
Supporting sources:
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/next-slice-brief-2026-04-04-quick-capture-to-draft.md`
- `docs/next-slice-brief-2026-04-04-draft-inspection-cues.md`

## Recommended slice

Add one bounded **Promote Draft** workflow that lets the user turn a quick-captured draft into a normal reusable prompt asset without hunting through the generic editing flow.

The point is simple:

> capture draft → inspect draft → promote to curated prompt

## Why this now

The last slices strengthened:
- **capture** via Quick Capture,
- **retrieve** via Recent Reopen,
- **inspect** via visible provenance cues.

The weakest remaining part of the core loop is now **normalize/refine into a durable asset**.

Today the user can capture something quickly, but there is still too much ambiguity between:
- a raw thing saved quickly,
- and a prompt trusted enough to keep and reuse.

This slice fixes that directly.

## User problem solved

After quick capture, the user should be able to do one short cleanup pass and end with:
- a proper title,
- better metadata,
- a non-draft state,
- a prompt that feels catalog-worthy.

Desired experience:

> open draft → promote it in one short flow → keep using it as a normal prompt

## Boundaries

### Do now
Implement exactly one narrow end-to-end workflow:

1. Add one visible **Promote Draft…** action for prompts marked as draft.
2. Show a compact promote/normalize flow using existing prompt data and existing save/update paths.
3. Focus only on a minimal curation set:
   - title
   - category
   - tags
   - source
   - short description / note
4. Reuse the existing prompt body instead of inventing a second content model.
5. Save the updated record through the existing persistence/update path.
6. On successful promotion, remove the draft marker from the existing quick-capture metadata so the prompt no longer presents as draft.
7. Keep the user in the current detail/editor flow after save.
8. Add one deterministic regression test covering the happy path.

### Strong defaults for v1
- Prefer existing prompt schema and existing editor/update seams.
- Prefer a compact dialog or focused editor mode over a new large screen.
- Prefer manual curation over AI-generated normalization.
- Preserve provenance where useful (`source`, `capture_method`) while removing only the active draft state.
- If a field is already acceptable, do not force over-validation.

## Do later
- duplicate/similar prompt checks during promotion
- AI title/tag/category suggestions
- bulk draft triage
- draft inbox / queues
- richer provenance timelines
- "when to use" guidance generation
- source-specific connectors/importers

## Acceptance checks

1. A quick-captured draft exposes one obvious **Promote Draft** action.
2. The user can complete promotion in one short flow using existing data.
3. The saved prompt remains in the existing catalog and detail/editor flow.
4. The draft cue disappears after promotion.
5. Existing provenance remains intact where appropriate.
6. The slice works without LiteLLM, Redis, web search, chains, or external services.
7. One deterministic regression test protects the happy path.
8. No unrelated surface area was expanded.

## Rollback

Rollback should be one isolated patch:
- remove the **Promote Draft** action,
- remove the narrow promote/normalize seam,
- remove the focused regression test,
- leave Quick Capture, Recent Reopen, detail view, and generic editing untouched.

## Anti-goals

- Do not redesign the whole prompt editor.
- Do not introduce a new prompt lifecycle system.
- Do not add bulk draft management.
- Do not add AI-assisted tagging, title, or category generation in this slice.
- Do not widen into duplicate detection yet.
- Do not touch chains, analytics, sharing, voice, or web enrichment.
- Do not add new persistence or schema sprawl unless absolutely required.
- Do not turn this into a generic ingestion platform.

## Suggested implementation posture

Keep it boring:
- one action,
- one short curation flow,
- one update path,
- one regression test.

The win is not sophistication.
The win is making **captured prompts become reusable prompt assets with very little friction**.

## Definition of done

The slice is done when:
- a quick-captured draft can be promoted in one short flow,
- the prompt no longer presents as draft afterward,
- the curated record remains visible in the current detail/editor flow,
- the implementation stays bounded,
- the test passes,
- nothing peripheral was expanded.
