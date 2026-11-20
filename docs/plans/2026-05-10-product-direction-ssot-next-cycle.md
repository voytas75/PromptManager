# PromptManager — Product Direction SSOT Next Cycle

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-11-20
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

Rating decision: do not select a rating/evaluation expansion as a next slice
without stronger evidence. Keep the existing optional single run rating and
aggregate, but expect sparse human input rather than designing a ranking around
it. For a quick personal return-to-this-prompt decision, use the existing
Favorite action/filter instead. Reconsider only if a repeatable evaluation method
is validated on real prompts and materially improves reuse/refinement choices;
see `docs/product-ssot.md` for the canonical semantics. This is a planning
posture, not a request to remove existing rating functionality.

## Current execution checkpoint

The product direction remains valid; the execution plan now starts with evidence-led selection rather than another predefined cue or feature.

- Most recently delivered bounded slice: [retrieval → inspect → reuse continuity](2026-11-18-retrieval-reuse-continuity.md), product commit `897260fea994cc8242005bca40cb253f21c93ecb`; its documentation correction is delivered as `bf389a9c22d9ac147349cb5ecb27334caa7b0443`.
- Exact documentation-SHA [Quality Gates 44820516127](https://github.com/voytas75/PromptManager/actions/runs/44820516127) succeeded: 1749 passed, 1 skipped; core coverage 82.25%. This is remote CI evidence, not native Windows GUI or provider qualification.
- Fresh local `.venv/bin/pyright --stats`: 273 files checked, 0 errors/warnings/informations under `pyrightconfig.json`. This configured scope excludes `cli/` and `scripts/`; no repository-wide typing cleanup is selected from this result.
- Code and focused test definitions already cover Unicode highlighting, filter/detail selection continuity, copy-body availability and the validate-path detail decision package. Do not schedule those delivered contracts again without a reproduced regression.
- No implementation successor is selected. Older plan headers saying `active` or `proposed` are not sufficient evidence of unfinished work; some ledgers retain such headers despite completed tasks and delivered closure sections. Their historical results must not be presented as fresh acceptance.
- Native Windows GUI acceptance remains unverified, not a confirmed defect or an automatic release blocker. Security work, live providers and cross-store crash/concurrency qualification remain separate scope decisions.

### Immediate sequence

1. **Bounded discovery — next planning action.** Inspect one end-to-end path on existing surfaces: search/filter → selected detail → Copy Prompt or Open in Workspace → edit/fork. Compare implementation and nearby tests with the intended stored-body, selected-UUID and lineage contracts. Record at most one consequential gap; classify it as reproduced defect, missing contract coverage, native-only uncertainty, or no real gap. Do not infer user hesitation or retrieval quality from unit tests alone.
2. **Select one successor only if evidence warrants it.** Prefer a reproducible continuity/asset-safety defect over another wording-only change. Specify exact files, acceptance criteria and provider-free regression scope in one new execution ledger. If no gap is found, close discovery without inventing implementation work; retain the current product baseline.
3. **Implement and qualify only the chosen scope after implementation authorization.** Use tests-first development and the existing environment. Update its ledger after each task; run affected/adjacent tests, Ruff and relevant Pyright checks. Imports and tests must use an inspected provider-free harness with isolated temporary state; no real catalog or provider credentials.
4. **Separate delivery from qualification.** Commit/push and exact-SHA CI verification follow explicit publication authorization. Native GUI or live-provider acceptance is added only when separately approved or necessary to resolve a concrete uncertainty in the selected capability.

### Decision check

Primary approach: bounded offline continuity discovery before any new feature. Strongest alternative: native Windows GUI acceptance of the delivered path. Switch to that alternative when an owner report or a reproduced host-specific discrepancy points to selection, clipboard or workspace behavior that offscreen/isolated tests cannot resolve. Neither path authorizes ranking redesign, new panels or broad technical cleanup.

Closed-slice history belongs in `docs/STATUS.md`, not here. This checkpoint summarizes selection evidence only; the linked ledger owns detailed historical receipts.

### September reconciliation

The 2026-09-20–22 CLI delivery set is closed supporting work, not a new product center. `catalog-check`, `prompt-validate`, `prompt-test`, `prompt-compare`, `prompt-lint`, `prompt-template-list`, and tag operations remain asset-supporting because they inspect, validate, compare, or organize existing prompt records without provider calls by default.

Do not extend this set automatically. Delivery-trust maintenance remains supporting work: lockfile reproducibility and the CI type-checking contract are known governance considerations, not newly authorized repairs or blockers for every slice. Select the next user-visible scope through the current execution checkpoint above. Provider-backed evaluation, dependency/impact graphs, broad chain work, and new integration surfaces remain separate decisions.

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

Selected supporting CLI contract: `docs/plans/2026-09-24-doctor-cli-contract-and-migration-v1.md`. The default `doctor` and its nested read-only diagnostics are delivered; `doctor embeddings --live` and `doctor analytics --live` are explicit, separately accepted backend probes, while the on-demand `doctor index` metadata-only slice was delivered as `63ca05d` with exact-SHA CI [36025044220](https://github.com/voytas75/PromptManager/actions/runs/36025044220) successful. Standalone legacy commands remain supported. This pointer does not reorder the asset-loop priorities above.

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
