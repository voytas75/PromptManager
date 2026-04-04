# PromptManager — Next Slice Brief

Date: 2026-04-04
Status: delivered
Slice name: Draft Inspection Cues
Primary source: `docs/product-boundary-ssot.md`
Supporting sources:
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/next-slice-brief-2026-04-04-quick-capture-to-draft.md`
- `docs/next-slice-brief-2026-04-04-recent-reopen.md`

## Recommended slice

Add one bounded **inspection clarity** pass that makes draft/provenance cues visible in the existing prompt detail view for recently captured or reopened prompts.

This slice should expose a small set of always-visible cues in the existing detail surface so the user can immediately understand:
- that a prompt is a draft,
- where it came from,
- and when it was last touched,
without opening the raw metadata panel.

## Why this now

The current mini-pack already strengthened:
- **capture** via Quick Capture,
- **retrieve/reopen** via Recent Reopen.

The weakest link now is **inspect**.

Today the important provenance signals created by Quick Capture (`source`, draft metadata) are technically preserved, but they are mostly buried in metadata views rather than visible in the primary detail surface.

That creates avoidable friction right after the new happy path:

> capture prompt → reopen prompt → still need to dig to understand what this thing is

This slice keeps the product centered on prompt assets by making the asset easier to inspect before any broader expansion.

## User problem solved

After capturing or reopening a prompt, the user should be able to tell at a glance:
- is this a draft or a normal prompt,
- where did it come from,
- how recent is it,
- whether it likely needs refinement.

The goal is simple:

> open prompt → immediately understand its status and provenance

## Boundaries

### Do now
Implement exactly one narrow inspection improvement in the existing detail view:

1. Surface a compact always-visible provenance/status row in the existing prompt detail panel.
2. Include only a minimal bounded cue set, using existing prompt fields/metadata:
   - source
   - last modified
   - draft status (from existing quick-capture metadata)
3. Keep the current metadata viewer intact as the deeper layer.
4. Add one deterministic test covering the rendered inspection cues for a captured draft prompt.

### Strong defaults for v1
- Prefer existing detail widgets over adding a new panel or dialog.
- Prefer plain textual cues/badges over decorative UI.
- Prefer existing fields and metadata (`source`, `last_modified`, `ext2`) over schema changes.
- If draft metadata is absent, show nothing extra rather than inventing new states.

## Do later
- richer provenance timelines
- edit-history summaries in the main detail view
- duplicate/similar prompt warnings on inspect
- “when to use” guidance generation
- source-type icons/connectors
- inbox/triage behavior for captured drafts

## Acceptance checks

1. A captured draft prompt shows a clear draft/provenance cue in the existing detail view.
2. A reopened prompt exposes source and last-modified information without requiring the metadata toggle.
3. The slice uses existing prompt data and does not add new persistence.
4. The slice works without LiteLLM, Redis, web search, chains, or external services.
5. One deterministic regression test protects the happy path.

## Rollback

Rollback should be one isolated patch:
- remove the added inspection cues from the detail surface,
- remove the focused regression test,
- leave capture, reopen, editor, and metadata views untouched.

## Anti-goals

- Do not redesign the whole detail view.
- Do not add a new editor workflow.
- Do not widen into analytics, favorites, pinning, or command palette work.
- Do not touch chains, sharing, voice, or web-enrichment behavior.
- Do not add new persistence or status models unless absolutely required.

## Suggested implementation posture

Keep the slice small and honest:
- one existing detail surface,
- one compact provenance/status row,
- one deterministic test,
- minimal doc update if needed.

The product win is not more UI.
The product win is letting the user understand a prompt asset immediately after capture or reopen.

## Definition of done

The slice is done when:
- draft/provenance cues are visible in the current detail flow,
- the user no longer needs the metadata panel for the basic “what is this?” answer,
- the implementation stays bounded,
- the test passes,
- nothing peripheral was expanded.

## Implementation note

Delivered on 2026-04-04 as a compact inspection line in the existing prompt detail widget.
Rendered cues use existing prompt data only:
- draft status from `ext2.capture_state` / `ext2.capture_method`
- `source`
- `last_modified`

The metadata panel remains the deeper layer; this slice only surfaced the minimal at-a-glance cues in the main detail flow.
