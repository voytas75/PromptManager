# PromptManager — Next Slice Brief

Date: 2026-04-04
Status: proposed
Slice name: Promote-time Similar Prompt Check v1
Primary source: `docs/product-boundary-ssot.md`
Supporting source: `docs/product-boundary-alignment-audit-2026-04-04.md`

## Task Brief

- Goal: Add one bounded similarity check to the existing Draft Promote flow so operators can notice likely existing prompt assets before promoting a captured draft as a new catalog item.
- Done when: The Draft Promote flow can show a small list of similar existing prompts and let the operator either continue promotion as a new asset or open one similar existing prompt, with focused regression coverage for the bounded happy paths.
- In scope: Only the existing Draft Promote flow, one lightweight similar-prompts presentation, and focused tests for no-match / some-match / continue-as-new / open-existing behavior.
- Out of scope: Full duplicate detection engine, auto-merge, auto-blocking promotion, batch import duplicate handling, analytics changes, chain/workspace/sharing/web-search work, or broad editor/catalog redesign.
- Context files: `docs/product-boundary-ssot.md`, `docs/product-boundary-alignment-audit-2026-04-04.md`, `docs/CHANGELOG.md`.
- Constraints: Keep the product center on prompt-asset quality; reuse existing retrieval/catalog capabilities where possible; prefer no schema migration; do not widen the public product contract; no commit/push.
- Expected output: Minimal patch summary, files changed, tests run, and any risk/uncertainty noted.
- Timebox: One bounded implementation slice.
- Rollback: Remove the similarity check from Draft Promote and revert the focused tests; no data migration rollback required.
- Escalate if: The smallest correct implementation requires a schema change, a broader retrieval refactor, or a blocking UX decision that widens the flow beyond this slice.

## Recommended slice

Add one **Promote-time Similar Prompt Check** inside the existing **Draft Promote** flow.

When the operator promotes a quick-captured draft into a reusable prompt asset, PromptManager should optionally surface a small list of the most similar existing prompts before the promotion finalizes.

The operator should then have exactly two safe paths:
- **Promote as new**
- **Open similar existing**

This slice should stay advisory only. It should not auto-merge, auto-block, or guess ownership decisions for the operator.

## Why this now

The 2026-04-04 alignment work already improved the core loop materially:
- Quick Capture
- Recent Reopen
- draft inspection cues
- Draft Promote / Normalize v1

That moved PromptManager closer to the SSOT product center.

The clearest remaining bounded gap is still **normalization quality at ingest/promotion time**, especially:
- recognizing likely duplicates,
- avoiding needless asset sprawl,
- helping the operator decide whether the draft is really new or just another version/near-copy.

This slice strengthens **Normalize + Inspect** without widening the product into a new subsystem.

## User problem solved

Today, an operator can quickly capture and promote a prompt draft, but the promote decision still has a blind spot:

> is this actually a new reusable prompt asset, or is it too close to something already in the catalog?

This slice adds a lightweight checkpoint at the exact moment where that decision matters most.

## Boundaries

### Do now
Implement exactly one narrow enhancement to the existing promote flow:

1. Reuse the current Draft Promote entry point.
2. Before final promotion completes, fetch a small list of the most similar existing prompts.
3. Present only lightweight operator-facing context for each result, such as:
   - title/name
   - category and/or tags
   - last modified
   - short similarity cue or concise identifying metadata
4. Provide exactly two primary actions:
   - **Promote as new**
   - **Open similar existing**
5. Keep the flow non-blocking when no useful matches are found.
6. Add focused deterministic regression coverage for the bounded cases.

### Strong defaults for v1
- Prefer existing semantic retrieval/catalog capabilities over new duplicate-specific infrastructure.
- Prefer a compact advisory UI inside the promote flow over a new dialog tree or major review screen.
- Prefer a short result list over exhaustive search output.
- Prefer operator judgment over automated dedupe decisions.
- If similarity confidence is weak or no matches exist, allow the current flow to continue with minimal extra friction.

## Do later
- explicit duplicate confidence scoring surfaced as a richer UX
- merge/fork guidance from similar results
- promote-time “replace/update existing” workflows
- batch import duplicate handling
- ingest-time source/context enforcement
- broader normalization assistants

## Acceptance checks

1. When promoting a draft, PromptManager can surface a small list of similar existing prompts if meaningful matches exist.
2. When no meaningful matches exist, the promote flow behaves essentially like it does today.
3. The operator can continue promotion as a new prompt asset without forced branching.
4. The operator can open one similar existing prompt directly from the flow.
5. The slice works without adding new external dependencies or widening the product surface.
6. Focused regression coverage exists for:
   - no matches
   - some matches shown
   - continue as new
   - open similar existing

## Rollback

Rollback should be one isolated patch:
- remove the similarity-check branch from Draft Promote,
- remove the lightweight advisory presentation,
- remove the focused regression tests,
- leave Quick Capture, Recent Reopen, inspection cues, and basic Draft Promote intact.

## Anti-goals

- Do not build a full dedupe engine.
- Do not auto-merge or auto-block promotion.
- Do not widen this into batch import logic.
- Do not redesign the whole prompt editor or catalog UX.
- Do not add analytics/dashboard work.
- Do not touch chains, sharing, voice, or web-enriched execution.
- Do not introduce schema sprawl unless absolutely required.

## Suggested implementation posture

Keep the slice small and honest:
- one bounded advisory check,
- one compact result presentation,
- two clear operator actions,
- focused tests,
- minimal doc/changelog updates if needed.

The product win is not clever automation.
The product win is reducing accidental prompt duplication while keeping the product centered on a high-quality local prompt asset base.

## Definition of done

The slice is done when:
- the existing Draft Promote flow can warn about likely similar prompts,
- the operator can still promote the draft as new with low friction,
- the operator can inspect/open a similar existing prompt when useful,
- the implementation stays bounded,
- the focused tests pass,
- no peripheral subsystem expands.
