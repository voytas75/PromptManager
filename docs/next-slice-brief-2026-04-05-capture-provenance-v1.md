# PromptManager — Next Slice Brief

Date: 2026-04-05
Status: delivered
Slice name: Capture Provenance v1
Primary source: `docs/product-boundary-ssot.md`
Supporting sources:
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/CHANGELOG.md`
- `README.md`

## Recommended slice

Add one bounded **Capture Provenance** improvement to the existing **Quick Capture** flow.

When an operator captures a draft prompt via Quick Capture, PromptManager should allow a simple source/provenance value to be recorded immediately and preserved through the existing normalize/inspect flow.

This slice should improve prompt-asset quality at capture time without widening into a full import platform, provenance taxonomy, or analytics track.

## Why this now

The 2026-04-04 and 2026-04-05 alignment work already materially strengthened the core loop:
- Quick Capture
- Recent Reopen
- draft inspection cues
- Draft Promote / Normalize v1
- Promote-time Similar Prompt Check v1
- Quick Reuse Handoff v1

That left a smaller but still important gap in **capture quality**:

> after capturing a prompt quickly, can the operator still tell where it came from later?

This slice strengthens **Capture + Inspect** together by ensuring a captured draft can carry simple provenance from the beginning.

## User problem solved

Without provenance, a captured prompt is structurally preserved but still partially context-poor.

The user may later reopen or promote a draft and still wonder:
- was this from a chat,
- from notes,
- from a script,
- or from some other ad-hoc source?

The goal is simple:

> capture prompt → keep where it came from → still see that later

## Boundaries

### Do now
Implement exactly one narrow upgrade in the existing Quick Capture flow:

1. Add one simple **Source / Provenance** input to the Quick Capture dialog.
2. Keep the field lightweight and optional.
3. Persist the entered value using the existing `prompt.source` storage path.
4. Preserve the stored source through the existing Promote flow.
5. Let the existing detail/inspection path show the source without adding a new provenance UI.
6. Add focused regression coverage for the bounded provenance happy paths.

### Strong defaults for v1
- Prefer one free-text field over a taxonomy system.
- Prefer the existing `source` field over any new storage model.
- Prefer current inspection cues over a new provenance panel.
- Prefer preserving fast capture over adding friction.
- If source is empty, fall back to the existing quick-capture storage marker.

## Do later
- richer source presets
- provenance-aware filtering/search
- stronger source/context pairing
- batch import provenance rules
- provenance timeline/history surfaces
- source normalization heuristics

## Acceptance checks

1. Quick Capture exposes a simple `Source / Provenance` input.
2. A captured draft stores the provided source using existing storage semantics.
3. If no source is provided, Quick Capture keeps the existing fallback source behavior.
4. Promoted prompts retain the captured source.
5. Existing inspect/detail cues still show source after promote.
6. No schema migration or new persistence subsystem is introduced.
7. Focused regression coverage protects quick-capture provenance input and inspect-path visibility.

## Rollback

Rollback should be one isolated patch:
- remove the `Source / Provenance` input,
- remove the helper/pass-through logic for the field,
- remove the focused provenance tests,
- leave the rest of Quick Capture, Promote, and inspect flows unchanged.

## Anti-goals

- Do not build a full import pipeline.
- Do not add a provenance taxonomy or source registry.
- Do not redesign Quick Capture.
- Do not redesign the detail view.
- Do not add provenance analytics.
- Do not widen into chains, sharing, web search, TTS, or execution changes.
- Do not turn this into a general ingest platform.

## Suggested implementation posture

Keep the slice small and honest:
- one optional field,
- one existing storage path,
- one preserved provenance signal,
- focused tests,
- no peripheral expansion.

The product win is not richer metadata for its own sake.
The product win is making captured prompt assets less context-blind later.

## Definition of done

The slice is done when:
- an operator can capture a prompt with simple provenance,
- that provenance stays attached through promote,
- the existing inspect path still surfaces it,
- the implementation stays bounded,
- the focused tests pass,
- nothing peripheral expands.

## Implementation note

Delivered on 2026-04-05 as a bounded `Source / Provenance` field in the existing Quick Capture dialog.

Implementation keeps the current storage model by persisting the entered value through the existing `prompt.source` path and preserves the existing fallback marker when the field is left empty.
Focused regression coverage protects:
- quick-capture provenance input
- fallback source resolution
- source visibility in the inspect path after promote
