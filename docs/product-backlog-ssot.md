# PromptManager — Product Backlog SSOT

Status: active
Owner: Wojtek / Prompt Manager Team

## Purpose

This file is the canonical backlog for PromptManager.

It translates the product boundary into a practical task order without using time-based planning.
It exists to answer one question:

> what should be worked on next, and what should explicitly not drive the product right now?

If this file conflicts with speculative ideas, side notes, or opportunistic feature expansion, this file wins unless explicitly superseded.

## Product center

PromptManager remains:

> local-first canonical home for prompt assets

The product is strongest when it improves the core loop:
- capture
- normalize
- retrieve
- inspect
- reuse
- refine

## Backlog rules

A backlog item is strong when it materially improves at least one of the following:
- faster capture
- cleaner normalization
- better retrieval confidence
- clearer inspection
- lower-friction reuse
- safer refinement or version clarity

A backlog item is weak when it mainly adds:
- surface area
- dashboard complexity
- execution-heavy behavior not needed for prompt reuse
- assistant-like novelty unrelated to prompt asset quality
- another subsystem that competes with the prompt catalog for product identity

## Already delivered, do not re-plan

The following bounded slices are already delivered and should not be planned again as new work:
- Quick Capture to Draft
- Recent Reopen
- Draft Promote / Normalize v1
- Reuse Polish v1
- Copy Prompt terminology consistency v1
- Copy Prompt docs cleanup v1
- Capture Provenance v1
- Usage Cue v1
- Retrieval Preview v1
- Similar Match Preview v1
- Context Lead Usage Cue v1
- Reuse Payload Tooltip v1
- Credible Source Cue v1

These are current baseline, not open backlog.

## Active backlog — ordered

### Priority 1 — Duplicate / Similar-on-Ingest Polish

Improve confidence during capture and promote when an incoming draft appears similar to existing prompt assets.

Scope:
- strengthen duplicate or similarity cues at the moment they help decisions most
- keep heuristics bounded and deterministic
- support operator judgment rather than auto-merging or auto-rewriting
- avoid turning ingest into a heavy classification pipeline

Why this matters:
- capture quality is part of the core product promise
- prompt libraries decay when near-duplicates accumulate silently
- better ingest quality reduces future retrieval friction

Delivered bounded slice under this priority:
- **Similar Match Preview v1** inside the existing Draft Promote advisory list
- adds one compact visible distinguishing cue for similar prompt matches using existing prompt data
- keeps the existing advisory flow and actions unchanged

Remaining work in this priority should only proceed if it still improves operator judgment at ingest time without widening into a duplicate-management subsystem.

### Priority 2 — Detail View Clarity Pass

Strengthen the inspect surface so the operator can decide quickly whether a prompt is the right asset to use.

Scope:
- improve bounded visibility of when-to-use, source, scenario, and context cues
- keep the existing detail flow as the main surface
- avoid introducing new panels, generated advice, or assistant behavior

Why this matters:
- inspect quality determines whether retrieval feels trustworthy
- a catalog is weaker when the user still has to read too much to understand what a prompt is for

Delivered bounded slice under this priority:
- **Context Lead Usage Cue v1** in the shared detail widget
- adds one bounded fallback so `When to use` can derive a compact cue from prompt body lead-in when saved cues are absent
- keeps the current detail flow and current labels unchanged

Remaining work in this priority should only proceed if it makes prompt fit easier to judge without turning detail view into a summary dashboard.

### Priority 3 — Reuse Friction Pass

Make the transition from finding a prompt to using it even more direct.

Scope:
- tighten Copy Prompt and adjacent reuse actions
- reduce unnecessary handoff friction
- preserve the current wording rule: `Copy Prompt`
- avoid expanding execution features while polishing reuse

Why this matters:
- PromptManager wins only if found prompts can be reused quickly
- fast reuse is more valuable than adding adjacent power features

Delivered bounded slice under this priority:
- **Reuse Payload Tooltip v1** in the shared detail widget
- explains what `Copy Prompt` and `Open in Workspace` will use in body-backed, description-fallback, and unavailable states
- keeps labels, actions, and layout unchanged

Remaining work in this priority should only proceed if it removes real reuse hesitation without creating a broader execution or helper framework.

### Priority 4 — Source / Provenance Leverage

Use existing provenance data more effectively across retrieval and inspection surfaces.

Scope:
- increase the practical value of the existing `source` field
- surface provenance where it helps retrieval confidence or inspect clarity
- avoid building a new provenance taxonomy, registry, or analytics track

Why this matters:
- provenance is already captured and should pay product value back
- source often helps distinguish similar prompts faster than title alone

Delivered bounded slice under this priority:
- **Credible Source Cue v1** in the shared detail widget
- filters low-signal technical source markers out of inspection cues while preserving credible provenance values
- reuses the same source-credibility logic as retrieval preview

Remaining work in this priority should only proceed if it increases source usefulness without widening into a provenance-management feature set.

### Priority 5 — Messy Input to Usable Draft

Make rough pasted prompt material easier to turn into a clean draft asset without broadening into a full import platform.

Scope:
- improve the path from raw text to usable draft
- preserve low-friction capture
- keep normalization bounded and transparent
- avoid broad multi-source ingestion architecture in this phase

Why this matters:
- the founding thesis assumes prompts come from messy scattered places
- capture loses value if rough source material still requires too much cleanup effort

### Priority 6 — Compare / Fork Clarity

Make prompt lineage more useful during decision-making.

Scope:
- help the operator understand what changed and why a fork exists
- improve confidence when choosing between similar versions
- avoid widening into a full diff platform or review workflow

Why this matters:
- refine quality depends on preserving useful history without making the catalog feel noisy
- version/fork support only matters if it helps a real reuse decision

### Priority 7 — README / Positioning Cleanup

Tighten product-facing language so the public story matches the actual product center.

Scope:
- make the prompt asset loop more dominant than adjacent subsystems
- reduce AI-workstation drift in product positioning
- keep docs aligned with the product boundary SSOT

Why this matters:
- product confusion often starts in wording before it shows up in roadmap drift
- if the README sells the wrong thing, future feature choices get worse

### Priority 8 — Core / Supporting / Freeze Classification Pass

Keep the roadmap honest by classifying major surfaces according to the product boundary.

Scope:
- mark what is core
- mark what is supporting
- mark what is frozen, later, or demoted unless directly justified
- use the result to guide future slice selection

Why this matters:
- PromptManager already shows scope expansion pressure
- classification reduces accidental roadmap drift

## Supporting backlog — only after higher priorities justify it

These can be worked on only when they clearly support the core loop above:
- light collections or favorites that improve retrieval speed
- light history cues such as recent or last-used signals
- bounded search clarity improvements such as subtle match highlighting
- bounded compare affordances between very similar prompt assets

These are supporting, not primary identity work.

## Freeze / demote unless directly justified

The following areas must not drive the product roadmap ahead of the active backlog above:
- prompt chains as a centerpiece
- broad analytics or dashboard expansion
- web-enriched execution as a product track
- voice / TTS as a major track
- broad sharing and collaboration expansion
- assistant-like features unrelated to prompt asset quality
- turning PromptManager into a general AI workstation
- turning PromptManager into an agent platform

These surfaces may remain in the product, but they should not outrank work that strengthens prompt asset management.

## Task selection rule

Before starting a new slice, ask:

> does this improve capture, normalize, retrieve, inspect, reuse, or refine more than the next highest backlog item?

If not, do not start it without an explicit decision.

## Default next implementation target

Unless a stronger reason appears, the next implementation target should be:

**Messy Input to Usable Draft**

Reason:
- Retrieval Preview v1, Similar Match Preview v1, Context Lead Usage Cue v1, Reuse Payload Tooltip v1, and Credible Source Cue v1 are already delivered
- retrieval, ingest confidence, inspect clarity, reuse clarity, and provenance clarity all improved in bounded steps
- the next strong product gain is helping rough pasted input become a usable draft with less cleanup friction

## Relationship to other SSOT files

Read these together when making product decisions:
- `docs/product-boundary-ssot.md`
- `docs/product-boundary-alignment-audit-2026-04-04.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- this file

Use this file for ordered backlog decisions.
Use the product boundary SSOT for product identity and scope discipline.
Use the restart brief for bounded implementation posture.
