# PromptManager — Product SSOT

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-09-02 (single product SSOT consolidation)

## Purpose

This file is the single source of truth for PromptManager product decisions.

It unifies product direction, scope boundary, backlog, and roadmap order in one place.
It exists to answer four questions in one place:

1. what PromptManager is
2. what PromptManager is not
3. what direction the product should evolve in
4. what should be worked on before lower-value expansion

If this file conflicts with older product SSOTs, side notes, speculative plans, or opportunistic feature work, this file wins unless explicitly superseded.

---

## One-sentence product definition

**PromptManager is a local-first system for capturing, organizing, retrieving, inspecting, reusing, and refining prompt assets, with an operational layer for trustworthy runs, routing, and diagnostics kept subordinate to that core.**

---

## Product center

PromptManager remains:

> local-first canonical home for prompt assets

This is the product center.
Everything else must either strengthen this center or remain secondary.

---

## North star

PromptManager should evolve into:

> local-first control plane for prompt assets and trustworthy prompt operations

This does not mean becoming a general AI workstation.
It means extending the prompt-asset core with a reliable operational layer built on the same product model.

Short version:

**assets first, operations second, automation third**

## Long-term strategic posture

PromptManager should continue to be positioned first as an operational memory and asset system for prompt and agent-workflow practice, not as a generic AI workspace.

Long-term preference order:
1. strengthen the prompt asset lifecycle,
2. strengthen reuse and inspection confidence,
3. add controlled execution, evaluation, and routing only when they reinforce the asset model,
4. avoid broadening into a general-purpose "AI for everything" surface.

Reading rule for future scope decisions:
- if a feature strengthens prompt assets as durable, reusable working objects, it fits the center;
- if it mostly adds ambient AI workspace breadth, it is secondary or out of scope.

---

## Core user

PromptManager is primarily for:
- a prompt-heavy AI power user
- a prompt engineer
- a developer or operator working with many prompts across many contexts
- a single-user, local-first operator who values speed, structure, recall, and durable ownership over collaboration ceremony

Typical needs:
- collect prompts from chats, notes, files, scripts, markdown, and experiments
- normalize them into durable reusable assets
- find them later in seconds
- understand when to use them
- reuse or refine them without losing history or context

---

## Product thesis

People who work heavily with prompts accumulate useful prompt and query assets across many fragmented places.
Those assets decay when they are:
- scattered
- inconsistently named
- weakly tagged
- hard to compare
- easy to forget
- expensive to rediscover

PromptManager exists to solve that exact problem first.

The operational layer matters only when it makes those assets easier to trust, run, compare, and improve.

---

## Primary job to be done

> I have a useful prompt or LLM query from somewhere, and I want to save it, organize it, find it later in seconds, understand whether it still fits, and use or refine it confidently.

---

## Core product loop

1. **Capture** — add or import a prompt or query from some source.
2. **Normalize** — give it title, body, description, tags, category, context, source, and notes.
3. **Retrieve** — find it later by text, metadata, recent use, or semantic similarity.
4. **Inspect** — understand what it does, when to use it, how trustworthy it is, and how it differs from related versions.
5. **Reuse** — copy, export, open in workspace, or use it as the basis for the next task.
6. **Refine** — improve it and preserve version or fork lineage.
7. **Run** — optionally execute it through a trustworthy operational layer when that helps evaluation or reuse.

If this loop is weak, the product is weak.
If this loop is coherent and trustworthy, the product is doing its job.

---

## Product posture

PromptManager is not just a prompt catalog.
PromptManager is also not a general AI workstation.

The target posture is:
- prompt assets as the primary product center
- execution and routing as an explicit operational layer
- diagnostics and configuration as first-class trust surfaces
- automation surfaces as controlled extensions of the same core

Constraint:
- every new layer must stay legible and subordinate to prompt asset management

---

## In scope — core

These areas define PromptManager and should receive the strongest product focus.

### 1. Prompt asset library
- canonical prompt records
- browsing, listing, grouping, filtering
- stable prompt identity

### 2. Capture and ingest quality
- quick capture
- paste/import from multiple sources
- bounded normalization on ingest
- duplicate and similar prompt awareness
- rough-input-to-usable-draft polish

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
- lineage / provenance signals

### 4. Retrieval and discovery
- full-text search
- metadata filtering
- semantic retrieval
- related/similar prompt discovery
- recent and bounded history cues where they materially help reuse

### 5. Inspection clarity
- readable detail view
- source/context visibility
- fit-to-use judgment
- version/fork clarity
- compact decision-support cues

### 6. Reuse and refinement
- copy prompt
- export prompt
- open in workspace
- safe editing
- versioning or fork lineage
- low-friction improvement workflow

### 7. Trust infrastructure
- deterministic settings resolution
- deterministic model/provider/embedding routing
- effective configuration visibility
- diagnostics that clearly distinguish `OK`, `WARN`, and `FAIL`
- explicit startup validation and fail-fast behavior for invalid critical state

---

## In scope — supporting but secondary

These are valuable only when they clearly support the core loop above.

### 1. Structured execution
Allowed when it helps answer:
- does this prompt still work
- what result did it produce
- which version is better
- should I reuse, refine, fork, or replace it

Execution is support for prompt assets, not the product center.

### 2. Recorded runs
- visible run inputs and outputs
- run history tied to prompt assets
- basic run provenance
- baseline vs candidate posture for prompt refinement
- bounded comparison between runs

### 3. Light analytics
- usage count
- last used
- basic quality/rating
- bounded token/cost/runtime signals when already available

Analytics should support curation, trust, and decision-making, not become a dashboard-first detour.

### 4. Automation surfaces
- stronger CLI workflows
- exportable or scriptable run operations
- API/internal service boundaries where justified
- bounded batch/repeatable execution paths

Automation is the third layer of the product, not a second product model.

### 5. Light collections or favorites
Useful when they materially improve retrieval speed and reduce clutter.

---

## Out of scope for the current product center

These may exist in bounded form, but they must not drive the roadmap ahead of core prompt-asset quality and trust infrastructure.

### Freeze or demote unless directly justified
- feature expansion mainly for UI breadth
- broad dashboarding for its own sake
- heavy prompt-chain expansion
- web research enrichment as a major product track
- voice / TTS as a major product track
- broad sharing/collaboration systems
- intent classification as a centerpiece
- scenario generation as a primary investment area
- assistant-like novelty unrelated to prompt asset quality
- turning PromptManager into a general AI workstation
- turning PromptManager into an agent orchestration platform
- integrations that bypass or complicate the core settings model
- automation that creates hidden parallel behavior outside the main product model

---

## Strategic product rule

When there is a conflict between:
- adding another visible feature surface
- and making configuration, routing, execution, or prompt asset quality more reliable and legible

prefer reliability, legibility, determinism, and product coherence.

PromptManager should grow by becoming more trustworthy and more operable, not mainly by becoming broader.

---

## Roadmap order

The intended product evolution order is:

### 1. Tighten retrieval-to-action confidence
Priority goes first to the operator path from finding a prompt to knowing what to do next.

Work in this stage should improve:
- retrieval and discovery legibility
- prompt-list confidence and scanability
- compact search-match reason clarity
- faster handoff from search or recent results into inspect, reuse, or refinement
- lower hesitation before opening detail or reusing immediately

Constraint:
- do not redesign ranking or retrieval architecture before this operator path is clearer.

### 2. Make inspect/detail a clear decision surface
After retrieval clarity, the product should make the detail surface answer the core operator questions with less ambiguity.

Work in this stage should improve:
- fit-to-use judgment
- evidence / provenance / next-action consistency
- compact decision-support cues
- comparison and validation readability
- handoff clarity from detail into reuse, workspace, or refinement

Constraint:
- prefer consistency and ordering improvements on existing surfaces over adding new panels or parallel guidance layers.

### 3. Tighten the reuse / refine / fork operating path
Once the user can find and understand the prompt faster, the next priority is reducing friction in the action path.

Work in this stage should improve:
- copy vs open-in-workspace clarity
- edit vs fork clarity
- lineage-preserving refinement decisions
- run-to-refine and inspect-to-refine continuity where it materially supports the asset loop
- low-friction movement from prompt asset to practical next action

Constraint:
- do not turn the editor, workspace, or workbench into a separate product center.

### 4. Make settings and routing deterministic
After the main asset loop is clearer, settings, provider resolution, model routing, and embedding resolution should remain fully legible and deterministic.

Required outcomes:
- one clear precedence model across JSON config, `.env`, runtime env, and defaults
- canonical handling of provider-specific aliases
- deterministic routing for fast, inference, and embedding paths
- no silent fallback behavior that hides the real effective state

Exit condition:
- the app can always answer what value is active, where it came from, and why it is being used

### 5. Make diagnostics a first-class trust layer
After deterministic configuration, the effective runtime state must become visible and operationally clear.

Required outcomes:
- effective settings view
- source-of-value visibility
- bounded startup and runtime warnings
- high-contrast health states for model access, embeddings, database, and vector store
- low-noise messaging that helps decisions

### 6. Mature execution into a structured run system
Execution should then mature as a structured run layer tied to prompt assets.

Required outcomes:
- reproducible runs
- visible run inputs and outputs
- run history tied to prompt assets
- practical run comparison
- baseline vs candidate posture for refinement

Constraint:
- execution must remain attached to prompt asset quality and reuse
- it must not redefine the product as an agent workbench

Current emphasis inside this stage:
- keep inspect/detail run evidence easier to read than to over-interpret,
- prefer wording/order fixes on existing summary seams before adding new cues,
- treat run-evidence legibility as a bounded continuation of the same asset-to-run-to-refine loop,
- for prompt chains, prefer explicit result contracts and stable operator/machine output semantics over heavier execution power.

### 7. Open automation surfaces deliberately
After the product is trustworthy for human operators, expose the same model through CLI/API automation.

Principle:
- GUI for interactive operation
- CLI/API for repeatable automation

Constraint:
- automation must expose the same product model, not create a shadow product

### 8. Add bounded evaluation and governance
Once runs and automation are solid, PromptManager can add practical evaluation and governance features.

Examples:
- simple quality comparison between prompt variants
- repeatable evaluation against selected inputs or scenarios
- visible provenance for prompt decisions
- clearer review posture around changes

Constraint:
- keep evaluation practical and decision-oriented
- avoid bloated benchmarking dashboards

### 9. Keep collaboration subordinate
Sharing and broader multi-user features may grow later, but they must remain subordinate to:
- local-first ownership
- prompt asset clarity
- deterministic operations
- trustworthy run provenance

---

## Product-layer readiness criteria

The roadmap order is a dependency order, not a calendar or a separate planning authority. A layer is ready to lead further work only when its outcome is visible in the existing product model:

1. **Trust and operability** — GUI and CLI expose the same core operational truth; users can see what is configured, optional, or blocking; invalid critical state is not silent.
2. **Prompt asset loop** — prompts move from rough input to reusable assets; users can find, judge, and reuse them faster than rediscovering them elsewhere.
3. **Structured operations** — runs are understandable and comparable; recorded evidence supports reuse, refinement, fork, or retirement decisions rather than creating noise.
4. **Automation** — headless workflows expose the same settings, diagnostics, and product model as interactive operation; no hidden parallel behavior is introduced.
5. **Later supporting surfaces** — sharing, collaboration, analytics, and convenience features remain additive and do not change the asset-first identity.

---

## Ordered product backlog

This is the ordered product backlog posture derived from the roadmap above.

### Priority 1 — Retrieval, inspect, and reuse confidence
Work that improves:
- stronger retrieval/discovery ergonomics, especially search-result legibility and faster reuse
- prompt-list confidence and clearer result-level action cues
- better inspect/detail decision support
- clearer fit-to-use judgment before action
- lower-friction reuse, refine, and fork decisions on existing seams
- faster continuity from search/recent into inspect and practical next action

Why this stays first:
- PromptManager wins or loses at the moment the operator must find the right prompt and decide what to do next.

### Priority 2 — Prompt-asset library quality
Work that improves:
- quick capture quality
- draft-to-asset promotion clarity
- duplicate and similarity judgment at ingest
- rough-input cleanup on the way to usable draft
- lower-friction CLI and import ergonomics for getting prompt assets into the catalog
- version/fork clarity where it materially supports later reuse

Why this remains high:
- prompt library quality is still the main product promise and the retrieval loop depends on strong asset inputs.

### Priority 3 — Trust infrastructure
Work that improves:
- settings precedence clarity
- routing determinism
- embedding resolution clarity
- startup validation
- effective config visibility
- diagnostics clarity
- reduction of ambiguous runtime noise

Why this comes next:
- operational trust should support the core asset loop without replacing it as the main product story.

### Priority 4 — Structured runs and run evidence
Work that improves:
- prompt-linked run records
- visible run provenance
- bounded comparison between runs
- reusable run context for refinement decisions
- bounded prompt-chain lifecycle completeness, inspectability, result semantics, and run evidence when kept subordinate to the prompt-asset model
- compact operator-facing run surfaces such as explicit final output vs final summary and deterministic machine-readable exports/results

### Priority 5 — Asset-to-run-to-refine coherence
Work that improves:
- inspect-to-run handoff
- run-result-to-revision decision flow
- clear reuse vs refine vs fork judgment after validation evidence exists
- compact inspect/detail cues that avoid repeating the same recommendation twice
- workbench continuity only where it clearly serves the prompt asset loop

### Priority 6 — Automation surfaces
Work that improves:
- CLI execution and inspection workflows
- scriptable run/export paths
- bounded API/service access where justified

### Priority 7 — Evaluation and governance
Work that improves:
- disciplined prompt comparison
- repeatable evaluation
- clear review posture and provenance-backed decisions

### Priority 8 — Collaboration and later supporting surfaces
Only after higher priorities clearly justify it.

---

## Default task selection rule

Before starting a new product slice, ask:

1. Does this materially improve capture, normalize, retrieve, inspect, reuse, refine, or trustworthy run support?
2. Does it make settings, routing, or execution more deterministic?
3. Does it make the effective state easier to understand and verify?
4. Does it strengthen the asset-to-run-to-refine loop?
5. Does it preserve the asset-first identity?
6. Does it avoid creating a second hidden product model?

If the answer is mostly no, it should not outrank the backlog above.

---

## Communication rule for README, roadmap, and public positioning

README, roadmap, and product communication should first sell:
- prompt assets
- retrieval
- inspect
- reuse
- refinement
- local-first ownership

Execution, routing, diagnostics, evaluation, and automation should be communicated as:
- supporting operational layers
- trust and decision infrastructure
- controlled extensions of the same product

Avoid describing PromptManager as:
- all-in-one AI studio
- agent platform
- general AI workstation
- broad LLM operations cockpit as the primary identity

---

## Decision summary

If there is ever a doubt whether PromptManager is:
- a prompt catalog,
- a workbench,
- an operations tool,
- or an automation surface,

the governing answer is:

**PromptManager is first a canonical home for prompt assets, then a decision-support system for finding, understanding, reusing, and refining them confidently, then a trustworthy operational layer around them, and only later a controlled automation surface built on the same model.**
