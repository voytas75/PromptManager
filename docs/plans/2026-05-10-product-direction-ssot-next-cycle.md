# PromptManager — Product Direction SSOT Next Cycle

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-05-10
Canonical product SSOT: `docs/product-ssot.md`
Related prior execution ledger: `docs/plans/2026-04-25-roadmap-implementation-plan.md`
Related bounded feature ledger: `docs/plans/2026-05-06-prompt-chain-rollout-plan.md`
Active next-cycle execution ledger: `docs/plans/2026-05-10-detail-edit-vs-fork-clarity-roadmap.md`

## Purpose

This file is the single planning note for the next PromptManager product cycle after the currently completed trust-foundation, retrieval-confidence, and bounded prompt-chain clarity work.

It exists to answer four questions:

1. what the next product cycle should optimize for,
2. what should not drive the next cycle,
3. which execution order best fits the current repo reality,
4. what bounded slices should be preferred next.

If older plans suggest many parallel follow-ups, this file wins for next-cycle prioritization until explicitly revised.

---

## One-line decision

**The next PromptManager cycle should strengthen retrieval, inspect, reuse, and refinement confidence around prompt assets, while treating trust surfaces and typing cleanup as supporting work that keeps those seams safe to evolve.**

Short rule:

**asset loop first, trust surfaces second, automation later**

---

## Current verified context

Confirmed from live repo checks on 2026-05-10:

- `docs/product-ssot.md` is active and coherent.
- `README.md` matches the asset-first product posture.
- repo working tree currently contains active bounded-slice changes rather than a clean checkpoint.
- test suite collection is healthy: `776 tests collected`.
- the just-finished `search/detail/reuse confidence` slice landed in code/tests and was closed in its execution ledger.
- `pyright --stats` is do weryfikacji in the current environment because `pyright` is not installed on PATH here.
- recent commits focus on workbench/refinement trust cues and prompt-chain semantics, not on a new product repositioning.
- bounded prompt-chain lifecycle surfaces already exist in code and docs (`prompt-chain-show --json`, `prompt-chain-run --json`, `prompt-chain-history`, explicit final output / summary semantics).

Operational reading:
- the product does **not** currently lack direction,
- the product **does** need a stricter next-cycle execution priority.

---

## What this cycle is trying to improve

The next cycle should make PromptManager better at the operator job:

> find the right prompt asset quickly, understand why it fits, decide what to do next, and reuse or refine it with less hesitation.

That means preferring work that improves:
- retrieval confidence,
- inspect/detail decision support,
- reuse/refine/fork clarity,
- prompt-to-run and run-to-refine continuity where it materially helps the asset loop,
- compact trust surfaces that reduce ambiguity.

---

## Recommended cycle order

### 1. Retrieval-to-action confidence
This is the primary next-cycle focus.

Reason:
- it most directly strengthens the product center,
- it now aligns with `docs/product-ssot.md` Roadmap order stage 1 and Priority 1,
- it builds on already-shipped retrieval/discovery and inspect/detail work instead of reopening settled trust-foundation slices.

Preferred seam types:
- prompt-list result legibility,
- search-match reason clarity,
- result-level action confidence before detail opens,
- faster handoff from search/recent into inspect, reuse, or refinement.

### 2. Inspect/detail as a decision surface
This is the second next-cycle focus.

Reason:
- the detail surface should answer whether a prompt fits, what evidence supports that judgment, and what the operator should do next,
- recent bounded slices already improved local trust/next-action seams and should now be treated as building blocks for a more coherent decision surface,
- this keeps the product centered on prompt-asset decisions rather than on broader feature expansion.

Preferred seam types:
- evidence / provenance / next-action consistency,
- fit-to-use judgment,
- compact comparison and validation readability,
- clearer detail-to-workspace or detail-to-refine handoff.

### 3. Reuse / refine / fork operating path
This is the third next-cycle focus.

Reason:
- after the user can find and understand a prompt faster, the next priority is reducing friction in the action path,
- PromptManager should make copy vs workspace, edit vs fork, and refine vs reuse easier to choose without opening a second product model.

Preferred seam types:
- copy vs open-in-workspace clarity,
- edit vs fork clarity,
- lineage-preserving refinement decisions,
- run-to-refine and inspect-to-refine continuity where it materially helps the asset loop.

### 4. Trust surfaces that support the asset loop
This is secondary but still important.

Reason:
- ambiguous routing/settings/runtime state weakens operator confidence,
- but trust work should support the core loop rather than replace it as the product story.

Preferred seam types:
- effective-state visibility,
- low-noise diagnostics,
- stronger status contrast,
- compact provenance for runtime decisions.

### 5. Structured runs and bounded execution ergonomics
This remains valid but should stay subordinate.

Reason:
- execution and prompt chains already have meaningful bounded surfaces,
- the main need is trust, inspectability, and lifecycle polish rather than a new execution-first product push.

Preferred seam types:
- validation/dry-run trust,
- result-consumption ergonomics,
- clarity of recent-run evidence,
- compatibility-safe model cleanup,
- bounded run evidence that feeds reuse/refine decisions.

### 6. Broader automation surfaces
Only after the previous layers remain coherent.

Reason:
- automation is useful only when it exposes the same product model cleanly,
- broadening it too early risks creating a shadow product.

---

## What must not drive the cycle

The next cycle should **not** be driven by:
- broad prompt-chain expansion,
- workflow-engine behavior,
- branching/loop/scheduler semantics,
- dashboard-first analytics,
- general AI workstation positioning,
- large collaboration/sharing expansion,
- automation breadth ahead of asset-loop clarity,
- repo-wide typing cleanup without a product seam justification.

Short rule:

**do not trade product center for feature breadth**

---

## Main bottleneck

The main current bottleneck is not product strategy.

The bottleneck is:

> keeping the next slices tightly focused on asset-loop confidence while paying down only the trust/typing debt that directly blocks those seams.

This matters because the repo is now in a state where:
- docs already say what PromptManager is,
- the risk is execution drift,
- and the easiest mistake would be to over-invest in secondary feature surfaces just because they are technically available.

---

## Recommended bounded-slice shape

The preferred current slice shape for this cycle is:

**do weryfikacji / choose explicitly after SSOT sync**

Current rule:
- do not treat any older active ledger pointer as authoritative by itself,
- choose the next bounded slice only if it clearly fits SSOT Priority 1 or Priority 2,
- prefer one existing operator-facing seam over a broader cross-surface rewrite.

Good constraints for the next slice:
- stay on existing prompt-list, detail, prompt-actions, or nearby presenter/controller seams,
- do not redesign retrieval ranking,
- do not add a new dashboard or persistence layer,
- prefer one clearly legible operator cue or one consistency fix over many weak cues,
- keep CLI/headless parity unchanged unless the cue clearly belongs in shared state.

Good candidate shapes under the new SSOT:
- one bounded prompt-list cue that shortens find -> decide,
- one bounded detail-surface consistency pass that aligns evidence / provenance / next action,
- one bounded reuse/refine/fork handoff cue on an existing action seam.

---

## Ordered next-slice candidates

The historical candidate ledger below remains as a delivery record for already-shipped bounded slices on this cycle.

Interpretation rule after the SSOT sync:
- treat these candidate blocks as delivered history, not as the current execution order,
- do not infer an active successor from them,
- choose the next bounded slice explicitly from the canonical `docs/product-ssot.md` priorities plus the updated cycle-order section above.

### Candidate 1 — Search-result action clarity beyond title match
**Status:** delivered previous slice

Goal:
- improve the operator’s confidence when search results match by source or scenario, not only by title.

Boundaries:
- no ranking changes,
- no retrieval model redesign,
- no persistence changes,
- no new workflow surfaces.

Delivered v1 baseline:
- `gui/prompt_list_model.py` now distinguishes title vs non-title list-local handoff confidence,
- `gui/prompt_list_delegate.py` now renders the bounded handoff cue as a visible row line,
- focused and nearby smoke tests cover model/delegate/presenter parity for this seam.

Next-step rule:
- do not reopen the shipped v1 cue family itself,
- choose the next bounded successor only if it stays on the same prompt-list/detail confidence seam without creating a broader retrieval layer.

Likely seams:
- `gui/prompt_list_model.py`
- `gui/prompt_list_delegate.py`
- `gui/prompt_list_presenter.py`
- nearby retrieval/detail tests

### Candidate 1 — Prompt list confidence and retrieval clarity
**Status:** delivered previous slice

Goal:
- strengthen trust and scanability at the find/choose boundary so operators can decide whether to inspect, reuse, or refine without opening every prompt.

Boundaries:
- prompt list and adjacent detail confidence cues only,
- no retrieval engine changes,
- no new ranking or search semantics.

Delivered v1 baseline:
- `gui/prompt_list_delegate.py` now renders bounded confidence/inspection cues with explicit typing-safe delegate seams,
- `gui/prompt_list_model.py` and prompt-list tests support that seam without broad retrieval churn.

Next-step rule:
- do not reopen the shipped prompt-list confidence cues unless a focused regression appears,
- choose the next bounded successor only if it stays on the same prompt-list/detail confidence seam without creating a broader retrieval layer.

Likely seams:
- `gui/prompt_list_model.py`
- `gui/prompt_list_delegate.py`
- `gui/prompt_list_presenter.py`
- nearby retrieval/detail tests

### Candidate 2 — Detail refine/fork action clarity
**Status:** delivered previous slice

Goal:
- reduce hesitation on refine/fork paths by making one concrete next move clearer on the existing detail/action seams.

Boundaries:
- compact action guidance only,
- no controller model rewrite,
- no new workspace workflow layer.

Delivered v1 baseline:
- `gui/widgets/prompt_detail_widget.py` already exposes decision, next-action, and workspace handoff surfaces on the detail view,
- focused widget coverage already proves validation-first and edit-before-reuse handoff cues on that surface,
- the missing fork-specific next-step cue has now been shipped on the same widget-local seam,
- the detail widget now keeps `Fork before editing` visible as a distinct next-action cue so the fork handoff can remain explicit instead of collapsing as a hidden duplicate.

Next-step rule:
- do not reopen the already-shipped validation-first, edit-before-reuse, or fork-before-editing handoff cues,
- choose a successor only if it stays on the same widget-local detail handoff surface,
- prefer a new compact cue only when it adds a genuinely missing operator next step.

Most recently delivered bounded successor ledger:
- `docs/plans/2026-05-10-detail-fork-handoff-clarity-roadmap.md`

### Candidate 3 — Active-seam typing cleanup
**Status:** delivered previous slice

Goal:
- reduce mechanical strict-typing risk in files most likely to be touched by active product slices.

Boundaries:
- choose smallest bounded files first,
- avoid broad repo churn,
- prefer files on retrieval/detail/execution seams.

Delivered v1 baseline:
- `gui/prompt_list_delegate.py` now has explicit index-like typing and local typed coercion around row preview/cue rendering,
- `gui/prompt_list_model.py` now exposes a typed `parent: QObject | None` constructor seam,
- `gui/prompt_actions_controller.py` now aligns its action-selection flow with current Qt typing and uses a public freshness helper boundary,
- `gui/workspace_history_controller.py` now uses typed metadata coercion helpers for run-summary and prompt-version extraction while keeping the freshness/next-action contract stable,
- `gui/dialogs/prompt_chains.py` now uses bounded typed helpers for status messaging, recent-run records, import payload coercion, and reasoning-payload traversal while keeping `Supporting summary` semantics intact,
- the bounded pyright verification packs for prompt-list, prompt-actions, workspace-history, and prompt-chains seams are green without changing retrieval/action behavior.

Next-step rule:
- do not reopen the shipped prompt-list delegate typing seam itself unless a focused regression appears,
- choose the next typing-cleanup slice only if it stays bounded to one active-seam file or one very small supporting contract,
- prefer the next smallest active seam over broad repo-wide cleanup.

Active bounded ledger at delivery time:
- `docs/plans/2026-05-10-prompt-list-delegate-typing-cleanup-roadmap.md`

### Candidate 4 — Prompt-chain ergonomics follow-up
**Status:** delivered previous slice

Goal:
- continue bounded prompt-chain lifecycle polish only after the main asset-loop slice is chosen or completed.

Boundaries:
- no engine expansion,
- no chain-first product shift,
- only clarity/inspectability/validation/consumption improvements.

Delivered v1 baseline:
- `gui/dialogs/prompt_chains.py` now frames the secondary result block as `Supporting summary`,
- focused prompt-chain dialog coverage now proves the summary reads as supporting context while `Final output` remains primary,
- the bounded verification pack stayed GUI-local without changing CLI/backend contracts.

Next-step rule:
- do not reopen the shipped `Supporting summary` wording seam itself,
- choose a prompt-chain successor only if it stays on one existing ergonomics seam such as validation clarity or inspectability,
- do not let prompt-chain work displace the prompt-asset product center.

Active bounded ledger:
- `docs/plans/2026-05-10-prompt-chain-result-decision-clarity-roadmap.md`

### Candidate 5 — Workspace compare-readiness clarity
**Status:** delivered previous slice

Goal:
- reduce hesitation on compare decisions by making one missing-baseline evidence path more explicit on the existing workspace-history trust surface.

Boundaries:
- one existing workspace-history controller seam,
- no prompt-detail handoff rewrite,
- no history model or workflow expansion.

Delivered v1 baseline:
- `gui/workspace_history_controller.py` now keeps the missing-baseline evidence path explicit without touching rating/duration/freshness branches,
- `Evidence: no comparable baseline yet` now maps to `Run a different prompt version before comparing`,
- focused workspace-history coverage proves the compare-readiness summary still reads `Comparison readiness: no baseline yet` while the operator action is more specific.

Next-step rule:
- do not reopen the shipped missing-baseline wording seam itself unless a focused regression appears,
- choose another trust-surface successor only if it stays on one existing evidence/decision seam,
- prefer the next most consequential evidence gap over broader history UX changes.

Most recently delivered bounded successor ledger:
- `docs/plans/2026-05-10-workspace-compare-readiness-clarity-roadmap.md`

### Candidate 6 — Workspace compare-rating clarity
**Status:** delivered previous slice

Goal:
- reduce hesitation on compare decisions by making the missing-rating evidence path more explicit on the existing workspace-history trust surface.

Boundaries:
- one existing workspace-history controller seam,
- no prompt-detail handoff rewrite,
- no history model or workflow expansion.

Delivered v1 baseline:
- `gui/workspace_history_controller.py` now keeps the missing-rating evidence path explicit without touching missing-baseline, missing-duration, or freshness branches,
- `Evidence: missing rating for comparison` now maps to `Add ratings to both runs before comparing`,
- focused workspace-history coverage proves the compare-readiness summary still reads `Comparison readiness: limited` while the operator action is more specific.

Next-step rule:
- do not reopen the shipped missing-rating wording seam itself unless a focused regression appears,
- choose another trust-surface successor only if it stays on one existing evidence/decision seam,
- prefer the next most consequential evidence gap over broader history UX changes.

Most recently delivered bounded successor ledger:
- `docs/plans/2026-05-10-workspace-compare-rating-clarity-roadmap.md`

### Candidate 7 — Workspace compare-duration clarity
**Status:** delivered previous slice

Goal:
- reduce hesitation on compare decisions by making the missing-duration evidence path more explicit on the existing workspace-history trust surface.

Boundaries:
- one existing workspace-history controller seam,
- no prompt-detail handoff rewrite,
- no history model or workflow expansion.

Delivered v1 baseline:
- `gui/workspace_history_controller.py` now keeps the missing-duration evidence path explicit without touching missing-baseline, missing-rating, or freshness branches,
- `Evidence: missing duration for comparison` now maps to `Run both versions again before comparing`,
- focused workspace-history coverage proves the compare-readiness summary still reads `Comparison readiness: limited` while the operator action is more specific.

Next-step rule:
- do not reopen the shipped missing-duration wording seam itself unless a focused regression appears,
- choose another trust-surface successor only if it stays on one existing evidence/decision seam,
- prefer the next most consequential evidence gap over broader history UX changes.

Most recently delivered bounded successor ledger:
- `docs/plans/2026-05-10-workspace-compare-duration-clarity-roadmap.md`

### Candidate 8 — Workspace one-run action clarity
**Status:** delivered previous slice

Goal:
- reduce hesitation on reuse decisions by making the recent single-run evidence path more actionable on the existing workspace-history trust surface.

Boundaries:
- one existing workspace-history controller seam,
- no prompt-detail handoff rewrite,
- no history model or workflow expansion.

Delivered v1 baseline:
- `gui/workspace_history_controller.py` now keeps the recent single-run branch action-oriented without touching stale single-run behavior, compare-readiness branches, or freshness wording,
- `Evidence: only one run available` now maps to `Run one more time before reusing` when freshness is recent,
- focused workspace-history coverage proves the limited-evidence provenance still reads `Based on limited run evidence` while the operator action is more specific.

Next-step rule:
- do not reopen the shipped recent one-run wording seam itself unless a focused regression appears,
- choose another trust-surface successor only if it stays on one existing evidence/decision seam,
- prefer the next most consequential evidence gap over broader history UX changes.

Most recently delivered bounded successor ledger:
- `docs/plans/2026-05-10-workspace-one-run-action-clarity-roadmap.md`

---

## Selection rule for the next implementation slice

Before starting a slice, confirm:

1. does it strengthen capture, retrieval, inspect, reuse, refine, or trustworthy run support around prompt assets?
2. does it improve operator confidence more than it increases feature breadth?
3. does it stay within one existing seam instead of opening a parallel product model?
4. if it touches trust or typing debt, does it directly support the active product seam?
5. does it avoid reopening work already marked delivered in current docs and tests?

If the answer is mostly no, it should not lead the next cycle.

---

## Documentation rule for this cycle

For the next cycle:
- keep `docs/product-ssot.md` as the canonical product truth,
- use this file as the direction note for next-cycle prioritization,
- open a fresh execution ledger only for the chosen bounded slice,
- do not rewrite broad product SSOT unless product truth changes,
- do not create multiple competing “next cycle” plans.

The most recently delivered bounded slice is tracked in:
- `docs/plans/2026-05-10-workspace-one-run-action-clarity-roadmap.md`

---

## Decision summary

If there is doubt what PromptManager should do next, the governing answer is:

**Strengthen the find -> understand -> decide -> reuse/refine loop around prompt assets, support it with compact trust surfaces, and treat typing cleanup and prompt-chain work as bounded enablers rather than the main product story.**
