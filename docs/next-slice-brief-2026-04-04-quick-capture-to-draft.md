# PromptManager — Next Slice Brief

Date: 2026-04-04
Status: delivered
Slice name: Quick Capture to Draft
Primary source: `docs/product-boundary-ssot.md`
Supporting source: `docs/product-boundary-alignment-audit-2026-04-04.md`

## Recommended slice

Add one **Quick Capture** happy-path workflow that lets the user paste a raw prompt or LLM query from any scattered source and save it immediately as a **draft prompt record** in the existing catalog.

This slice should end by opening the newly created draft in the existing prompt detail/editor flow so the user can inspect and refine it.

## Why this now

This is the strongest small slice because it directly reinforces the original mission of PromptManager:
- gather prompts and LLM queries from scattered sources,
- bring them into one durable prompt base,
- normalize them enough to become reusable assets.

It also addresses the clearest gap identified in the alignment audit:
- the product already has a strong library/retrieval/editing core,
- but **capture/import is not yet the visible product center**.

## User problem solved

Today, a useful prompt found in a chat, note, script, markdown file, or ad hoc experiment still has too much friction before it becomes a durable asset in PromptManager.

This slice should make the first step almost boring:

> paste raw prompt/query → save as draft → inspect/refine later

## Boundaries

### Do now
Implement exactly one narrow end-to-end workflow:

1. Add one **Quick Capture** entry point in the existing GUI.
2. Allow the user to paste raw prompt/query text.
3. Allow a minimal metadata set only:
   - title/name
   - raw prompt/query body
   - optional source label
   - optional tags
   - optional short note/description
4. Save the result as a **draft prompt record** using the existing persistence path.
5. Open the newly created record in the existing detail/editor view.
6. Add one deterministic test for the capture-to-record conversion or save flow.

### Strong defaults for v1
- Prefer existing prompt schema and storage.
- Prefer a simple modal/dialog or compact panel over new large surfaces.
- Prefer manual metadata entry over speculative AI-generated normalization.
- If title is omitted, use one simple deterministic fallback, e.g. first non-empty line trimmed to a safe length.

## Do later
- duplicate/similar prompt detection on capture
- multi-file or directory imports
- browser/chat/connectors
- OCR or document extraction
- AI-generated normalization/tagging/title suggestion
- bulk ingest workflows
- capture queues/inboxes

## Acceptance checks

1. A user can create a draft prompt from pasted raw text through one obvious GUI entry point.
2. The saved item lands in the existing prompt catalog using the current persistence layer.
3. The newly created item opens in the existing detail/editor flow without extra navigation.
4. The flow works without LiteLLM, Redis, web search, chains, or external services.
5. One deterministic regression test protects the happy path.

## Rollback

Rollback should be one isolated patch:
- remove the Quick Capture UI entry point,
- remove the narrow capture/save seam,
- remove the focused regression test,
- leave the rest of the catalog/editor/runtime untouched.

## Anti-goals

- Do not redesign the whole prompt editor.
- Do not add a second major workflow surface.
- Do not widen into generic import infrastructure.
- Do not add AI-assisted tagging, title generation, or classification in this slice.
- Do not bundle duplicate detection, retrieval changes, or analytics work.
- Do not touch chains, sharing, voice, or web-enrichment behavior.
- Do not introduce new persistence technology or schema sprawl unless absolutely required.

## Suggested implementation posture

Keep the slice small and honest:
- one entry point,
- one capture form,
- one save path,
- one deterministic test,
- minimal doc update if needed.

The product win is not sophistication.
The product win is reducing the friction between **finding a useful prompt somewhere** and **having it safely inside PromptManager**.

## Definition of done

The slice is done when:
- the user can capture one raw prompt/query into the catalog in a single short flow,
- the resulting record is visible and editable immediately,
- the implementation stays bounded,
- the test passes,
- nothing peripheral was expanded.

## Implementation note

Delivered on 2026-04-04 as a compact draft-capture flow wired from the main toolbar.
The slice uses existing prompt persistence and existing prompt fields, with draft state carried in `ext2` metadata.
The created draft is selected immediately and handed off into the existing editor flow.
