# PromptManager — Product Direction SSOT Next Cycle

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-05-17
Canonical product SSOT: `docs/product-ssot.md`
Status/history sink: `docs/STATUS.md`

## Purpose

This file is the single active near-term planning note for PromptManager.

It should stay short.
It exists to answer only:
1. what the next product cycle should optimize for,
2. what should not drive the cycle,
3. what order best fits the current SSOT.

If older plans or ledgers suggest many parallel follow-ups, this file wins for near-term prioritization until explicitly revised.

---

## One-line decision

**The next PromptManager cycle should strengthen retrieval, inspect, reuse, and refinement confidence around prompt assets, while treating trust surfaces as supporting work rather than a second product story.**

Short rule:

**asset loop first, trust surfaces second, automation later**

---

## Current planning posture

PromptManager does not currently lack product direction.
The main risk is execution drift into breadth.

So the near-term plan should stay focused on:
- operator confidence,
- decision clarity,
- low-friction reuse/refinement,
- compact trust support.

Closed-slice history belongs in `docs/STATUS.md`, not here.

---

## Recommended near-term order

### 1. Tighten retrieval-to-action confidence
This remains the first priority.

Focus on bounded work that improves:
- prompt-list scanability,
- search/result legibility,
- result-level action confidence,
- faster handoff from search/recent into inspect, reuse, or refinement.

Constraint:
- do not redesign retrieval ranking or open a second retrieval model.

### 2. Make inspect/detail a clearer decision surface
This remains the second priority.

Focus on bounded work that improves:
- fit-to-use judgment,
- evidence / provenance / next-action consistency,
- compact comparison and validation readability,
- clearer detail-side action handoff.

Constraint:
- prefer consistency on existing surfaces over new panels or parallel guidance layers.

### 3. Tighten the reuse / refine / fork operating path
This remains the third priority.

Focus on bounded work that improves:
- copy vs workspace clarity,
- edit vs fork clarity,
- lineage-preserving refinement decisions,
- inspect-to-refine or run-to-refine continuity only where it directly helps the asset loop.

Constraint:
- do not turn workspace or execution into a separate product center.

### 4. Do only trust-surface work that directly supports the asset loop
This is supporting work, not the lead story.

Focus on bounded work that improves:
- effective-state visibility,
- deterministic settings/routing understanding,
- low-noise diagnostics,
- compact runtime provenance.

Constraint:
- trust work should support the active operator seam, not replace it as the roadmap center.

### 5. Keep structured runs and prompt-chain work subordinate
This remains valid only when it improves asset-to-run-to-refine clarity.

Allowed focus:
- result-consumption ergonomics,
- bounded run evidence readability,
- prompt-chain/result semantics only where they feed reuse/refine decisions.

Constraint:
- no execution-first or prompt-chain-first cycle.

---

## What must not drive the cycle

The next cycle should not be driven by:
- broad prompt-chain expansion,
- workflow-engine behavior,
- dashboard-first analytics,
- general AI workstation positioning,
- collaboration/sharing breadth,
- automation breadth ahead of asset-loop clarity,
- repo-wide typing cleanup without product-seam justification.

Short rule:

**do not trade product center for feature breadth**

---

## Slice selection rule

Before starting a slice, confirm:
1. it strengthens capture, retrieve, inspect, reuse, refine, or trustworthy run support around prompt assets,
2. it improves operator confidence more than it increases breadth,
3. it stays within one existing seam,
4. it supports the current product center rather than opening a shadow product,
5. it does not reopen delivered work without a verified reason.

If the answer is mostly no, it should not lead the cycle.

---

## Documentation rule

For this cycle:
- keep `docs/product-ssot.md` as the canonical product truth,
- keep this file as the single active near-term plan,
- keep delivered history in `docs/STATUS.md`,
- open a fresh execution ledger only for the chosen bounded slice,
- do not create competing plan files.

---

## Decision summary

If there is doubt what PromptManager should do next, the governing answer is:

**Strengthen the find -> understand -> decide -> reuse/refine loop around prompt assets, support it with compact trust surfaces, and avoid drifting into a broader AI workstation.**
