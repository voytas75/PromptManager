# PromptManager — Product Boundary SSOT

Status: active
Owner: Wojtek / Prompt Manager Team
Last updated: 2026-04-04

## Purpose

PromptManager is a **local-first canonical home for prompt assets**.

Its primary job is to help a single operator collect prompts and LLM queries from scattered sources, normalize them into a reusable catalog, retrieve them quickly, inspect their context, and reuse or refine them without losing structure or history.

This product is **not** primarily an agent platform, a general-purpose AI workstation, or a workflow automation studio.

## Product thesis

People who work heavily with prompts quickly accumulate useful prompt/query assets across chat transcripts, notes, files, scripts, markdown documents, and experiments.

Those assets become hard to reuse because they are:
- scattered,
- inconsistently named,
- weakly tagged,
- hard to compare,
- easy to forget,
- expensive to rediscover.

PromptManager exists to solve that exact problem.

## Core user

Single-user, local-first prompt-heavy operator.

Typical profile:
- actively works with LLMs,
- collects prompts from many places,
- wants one durable prompt base,
- values speed, structure, and recall over collaboration ceremony,
- prefers local control over cloud-first product assumptions.

## Primary job to be done

> I have a useful prompt or LLM query from somewhere, and I want to save it, organize it, find it later in seconds, and use it again confidently.

## Supporting jobs

- Understand what a prompt is for.
- See when to use it.
- Compare or refine versions without losing the older one.
- Find similar prompts by meaning, not just by exact words.
- Keep one prompt library instead of scattered fragments across tools.

## Core product loop

1. **Capture** — add/import a prompt or query from any source.
2. **Normalize** — give it title, body, description, tags, category, context, source, and optional notes.
3. **Retrieve** — find it later by text, metadata, or semantic similarity.
4. **Inspect** — understand what it does, when to use it, and how it differs from related versions.
5. **Reuse** — copy, export, or run it with minimal friction.
6. **Refine** — improve it and keep version/fork lineage.

If this loop is weak, the product is weak.
If this loop is excellent, the product is doing its job.

## In scope — core

These areas define PromptManager and should receive the strongest product focus.

### 1. Prompt library
- canonical prompt records
- browsing, listing, grouping, filtering
- stable prompt identity

### 2. Prompt capture and import
- quick add
- paste/import from multiple sources
- normalization on ingest
- duplicate/similar prompt awareness

### 3. Metadata and structure
- title
- prompt body / query text
- description
- category
- tags
- source
- context / when to use
- notes
- status / active flag

### 4. Search and retrieval
- full-text search
- metadata filtering
- semantic search
- related/similar prompt discovery

### 5. Prompt inspection
- readable detail view
- source/context visibility
- example input/output where useful
- related prompts / lineage

### 6. Prompt editing and refinement
- safe editing
- versioning or fork lineage
- history visibility
- low-friction improvement workflow

### 7. Quick reuse
- copy prompt
- export prompt
- open in workspace
- optional lightweight execution path

## In scope — supporting but secondary

These are useful only if they clearly support the core loop above.

### 1. Lightweight execution
Allowed when it helps answer:
- does this prompt still work,
- what result did it produce,
- which version is better.

Execution is a support feature, not the product center.

### 2. Light history
- recent prompts
- last used
- last modified
- favorites / saved sets

### 3. Light analytics
- usage count
- last used
- basic quality/rating
- basic token/cost signals if already available

Analytics should support curation and reuse, not become a dashboard-first detour.

### 4. Collections / favorites
Useful when they make retrieval faster and reduce clutter.

## Out of scope for current product center

These may exist, but they should not drive the product roadmap ahead of core prompt-base quality.

### De-prioritized / later
- heavy prompt-chain expansion
- web research enrichment as a major product track
- voice/TTS as a major track
- broad sharing/collaboration systems
- intent classifier as a product centerpiece
- scenario generation as a primary investment area
- turning PromptManager into a general AI workstation
- turning PromptManager into an agent orchestration platform

## Product boundary rule

When evaluating a feature, ask:

> Does this make capture, normalization, retrieval, inspection, reuse, or refinement of prompt assets materially better?

If **yes**, it likely belongs.
If **not**, it is secondary, later, or out of scope.

## Current drift risk

PromptManager already shows signs of scope expansion into:
- execution-heavy workspace behavior,
- analytics expansion,
- prompt chains,
- web-enriched workflows,
- assistant-like features around prompts.

Those features are not automatically wrong.
They become wrong when they degrade focus on the core prompt asset loop.

## Product positioning

Recommended positioning:

> PromptManager is a local-first prompt catalog and reuse workspace.

Short version:

> Canonical home for prompt assets.

Avoid positioning it as:
- all-in-one AI studio,
- multi-agent platform,
- broad LLM operations cockpit.

## Prioritization rule

### Build now
- capture/import quality
- prompt object quality
- retrieval ergonomics
- semantic recall quality
- metadata clarity
- detail/inspection UX
- low-friction reuse
- version/fork clarity

### Build carefully
- basic execution
- light history
- light analytics
- collections/favorites

### Freeze or demote unless directly justified
- large new chain investments
- big dashboard work
- non-essential integrations
- novelty UX that does not improve prompt asset management

## Non-goals

PromptManager should not try to be, right now:
- the best place to orchestrate complex agent workflows,
- the best place to browse the web through LLM tooling,
- the best place to build general AI productivity flows,
- a collaboration platform for teams,
- a voice-first AI client.

## Immediate implications for product decisions

1. Prefer improvements to prompt ingestion over new peripheral modules.
2. Prefer improvements to retrieval quality over decorative analytics.
3. Prefer clearer prompt objects over more feature surfaces.
4. Prefer reuse speed over feature breadth.
5. Treat execution as support for the catalog, not as the main identity.

## Suggested near-term evaluation checklist

A proposed change is strong when it improves at least one of these:
- faster capture,
- cleaner normalization,
- better metadata,
- better semantic or exact retrieval,
- easier inspection,
- safer refinement/versioning,
- lower-friction reuse.

A proposed change is weak when it mainly adds:
- side workflows,
- extra UI complexity,
- dashboards without decision value,
- assistant behavior unrelated to prompt asset quality.

## Recommended next planning slice

Define and validate **Core Product Boundary v1** across the existing implementation:
- what is truly core today,
- what is supporting,
- what should be frozen or demoted,
- where the current UI and roadmap drift from the product center.

This SSOT is the canonical source for those decisions unless explicitly superseded.
