# PromptManager — Product Boundary Alignment Audit

Date: 2026-04-04
Expected source: `docs/product-boundary-ssot.md`
Reviewer: main

## Verdict

**Partially aligned.**

PromptManager already contains a strong product core for prompt asset management, but the implemented and documented surface has widened beyond the new product center. The app is not off-track beyond recovery; it is better described as **core-capable but scope-expanded**.

## What matches the SSOT

### 1. Prompt library is real and central
Evidence in README and implementation strongly supports the core idea of a canonical prompt base:
- catalog/list/search/detail panes in GUI
- CRUD flows
- semantic retrieval
- prompt schema with canonical fields
- fork/version history

This is the strongest alignment point.

### 2. Retrieval is a first-class capability
The product already invests in:
- full-text-like discovery and category/tag filters
- semantic search via ChromaDB
- related prompt retrieval by meaning

This aligns directly with the SSOT requirement that prompt assets be quickly recoverable.

### 3. Metadata and prompt structure are substantial
The prompt object and surrounding UX already support structured prompt records rather than loose text blobs.
This matches the SSOT expectation that prompts be normalized into durable, inspectable assets.

### 4. Editing/refinement is supported
Prompt editing, refinement workflows, template preview, and version/fork handling fit the product center well.
These improve the **refine** stage of the core loop.

### 5. Local-first posture remains intact
SQLite + ChromaDB + optional Redis + optional LiteLLM still support the intended local-first posture.
This is strategically aligned with the single-user prompt-base thesis.

## What is missing relative to the SSOT

### 1. Capture/import is not yet the visible product center
The original product mission was to gather prompts and LLM queries from many scattered sources into one place.

Current implementation clearly supports catalog import/export, but the visible product story and feature emphasis lean more toward execution/workspace capability than toward **capture and normalization**.

What appears missing or under-emphasized:
- quick capture workflow from arbitrary pasted prompt/query text
- stronger ingest UX for scattered-source collection
- explicit duplicate/similar-on-ingest guidance in the product surface
- stronger source attribution as a front-and-center prompt asset property

### 2. The canonical prompt object is not yet the dominant narrative
The SSOT centers the product on the prompt as an asset.
The current README still markets many surrounding systems at comparable weight.

Missing product emphasis:
- prompt as primary object
- source, context, and reuse cues as part of the main story
- clearer “when to use this prompt” posture in product language

### 3. Reuse speed is not yet clearly the main success metric
The SSOT says the product wins when users can find and reuse prompts quickly.
The current docs describe many capabilities, but they do not yet sharpen the product around **low-friction retrieval and reuse** as the main benchmark.

### 4. Import breadth appears narrower than the mission suggests
Evidence reviewed points to catalog import/export centered around JSON payloads and prompt catalog utilities.
That is useful, but narrower than the founding narrative of absorbing prompts from many scattered source types.

This does not mean the repo is wrong.
It means the ingestion story is probably weaker than the product thesis.

## What drifted / widened beyond the product center

### 1. Execution workspace became very prominent
Execution is useful, but it now occupies too much product identity relative to the SSOT.

Signs of widening:
- dedicated execution workspace emphasis in README
- streaming/chat-style execution framing
- broad prompt-running ergonomics beyond lightweight validation/reuse support

This is the biggest drift risk because execution can easily become a second product inside the first one.

### 2. Analytics expanded beyond “light curation support”
The app includes:
- analytics dashboard
- token usage tracking
- cost breakdowns
- benchmark views
- execution analytics CLI
- embedding diagnostics and exports

Useful, yes.
But relative to the SSOT, this is already more than “light analytics”.
This is a widening track that can consume roadmap energy quickly.

### 3. Prompt chains are a clear scope expansion
Prompt chains move the product toward workflow automation/orchestration.
That is explicitly outside the current product center unless it directly improves prompt asset reuse.

Today this looks like a meaningful widened surface, not just a tiny supporting helper.

### 4. Web search enrichment pushes toward AI workstation territory
Web-enriched execution is interesting, but it is not central to prompt asset management.
As a roadmap driver, it risks moving PromptManager from “home for prompt assets” toward “general LLM operations desktop”.

### 5. Sharing is broader than the current single-user core requires
ShareText, Rentry, and PrivateBin are legitimate features, but they are not central to the current product boundary.
They are supporting at best, and potentially distracting if expanded further.

### 6. Voice/TTS is peripheral
Voice playback is the kind of feature that may be pleasant but is hard to justify as core to prompt capture, retrieval, inspection, reuse, or refinement.
It should be treated as clearly secondary.

### 7. Intent classification and scenario generation sit near the boundary
These can support discovery and “when to use” guidance, so they are not automatically drift.
But as standalone investment areas they pull the product toward assistant behavior rather than prompt asset management.

## What is unverified

These points require live UX review or user observation rather than repo/docs inspection alone.

### 1. Whether the core loop is actually friction-light in practice
The codebase and docs suggest the pieces exist.
What is still unverified is whether a user can really do this smoothly:
- capture a prompt,
- normalize it,
- find it a week later,
- reuse it in seconds.

### 2. Whether the current UI prioritizes retrieval over feature breadth
The repository shows many features.
A live product pass is still needed to confirm whether the UI hierarchy reinforces the SSOT or overwhelms it.

### 3. Whether advanced features are already causing user confusion
Repo evidence shows expansion.
It does not prove whether users experience that as power or clutter.

## Recommended next action

### Recommended next slice
**Core Loop Alignment Pass v1**

A bounded next slice should not add new features.
It should align the existing product with the SSOT.

#### Do now
1. classify current surfaces into:
   - core
   - supporting
   - demote/later
2. tighten product-facing docs so README reflects the product center
3. identify the highest-friction gap in the core loop, likely one of:
   - capture/import,
   - normalize/metadata,
   - retrieval ergonomics,
   - inspection clarity,
   - quick reuse
4. choose exactly one small implementation slice from that gap

#### Do later
- bigger roadmap pruning
- chain strategy decision
- analytics reduction or refocusing
- workspace simplification strategy

## Suggested module-level classification

### Core
- prompt catalog / library
- prompt schema and metadata model
- semantic retrieval
- prompt editing/refinement
- version/fork/history for prompt assets
- catalog import/export foundation
- detail/inspection UX

### Supporting
- lightweight execution
- light history
- favorites/collections/categories
- template preview and validation
- prompt parts, if treated as reusable prompt assets rather than a separate system

### Demote / later / freeze unless directly justified
- prompt chains as a roadmap centerpiece
- broad analytics/dashboard expansion
- benchmark-heavy product emphasis
- web search enrichment as a major track
- sharing expansion
- voice/TTS
- assistant-like surface growth unrelated to prompt asset quality

## Sources reviewed

- `docs/product-boundary-ssot.md`
- `README.md`
- `docs/README-DEV.md`
- `core/catalog_importer.py`
- `core/execution.py`
- `core/analytics_dashboard.py`
- `core/intent_classifier.py`
- `core/scenario_generation.py`
- `core/sharing.py`
- `models/prompt_chain_model.py`
- `.memo/metadata.md`
- `.memo/state.md`
- `.memo/decisions.md`
- `.memo/knowledge.md`
