# PromptManager — Status

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-05-17
Canonical product SSOT: `docs/product-ssot.md`
Canonical near-term plan: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Purpose

This file is the compact status/history sink for PromptManager.

Use it to record:
- what has already landed,
- which slice families are considered closed by default,
- which execution ledger was most recently delivered,
- where to look before choosing the next bounded slice.

Do not use this file as a second product SSOT or a competing roadmap.

---

## Current product posture

PromptManager remains:

> local-first canonical home for prompt assets

Operational reading:
- assets first,
- operations second,
- automation later.

When in doubt, `docs/product-ssot.md` wins.

---

## What is already delivered enough to not plan again by default

### Core-loop base already landed
The following core-loop families are already in place and should not be re-planned from scratch unless a focused regression or new gap is verified:

- Quick Capture to Draft
- Recent Reopen
- Draft Promote / Normalize v1
- Reuse Polish v1
- Copy Prompt terminology consistency / docs cleanup
- Capture Provenance v1
- Usage Cue v1
- Retrieval Preview v1
- Similar Match Preview v1
- Context Lead Usage Cue v1
- Reuse Payload Tooltip v1
- Credible Source Cue v1
- Fence Unwrap v1
- Fork Baseline Clarity v1
- Fork Difference Cue v1
- Similarity Strength Cue v1
- Catalog Readability Typography v1
- Promote-time Likely Duplicate Cue v1
- Duplicate Reason Cue v1
- Prompt Label Strip v1
- Template Variable Cue v1
- Blockquote Unwrap v1
- Template Workspace Handoff Cue v1
- Search Match Highlight v1
- Source-Matched Preview Priority v1
- Search Error Specificity v1
- Edit Dialog Promote Draft Shortcut v1

Primary historical reference: `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

### Later bounded confidence/trust slices already landed
These later slice families are also treated as delivered history rather than active planning candidates by default:

- prompt-list confidence and retrieval clarity slices
- detail edit vs fork clarity slices
- prompt-list delegate typing cleanup slice
- prompt-chain result / handoff clarity slices already marked delivered
- workspace compare-readiness clarity
- workspace compare-rating clarity
- workspace compare-duration clarity
- workspace one-run action clarity

Primary historical reference: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

---

## Current planning rule

Before selecting the next slice, confirm:
1. it strengthens capture, retrieve, inspect, reuse, refine, or trustworthy run support around prompt assets,
2. it improves operator confidence more than it increases breadth,
3. it stays on one existing seam,
4. it does not reopen a delivered slice family without a verified reason,
5. it keeps PromptManager asset-first.

If not, it should not lead the next cycle.

---

## Most recent delivered execution ledger

Most recently delivered bounded execution ledger remembered in active planning docs:
- `docs/plans/2026-05-10-workspace-one-run-action-clarity-roadmap.md`

This is a status pointer only, not an instruction to continue that seam by default.

---

## Where to look before choosing the next slice

Read in this order:
1. `docs/product-ssot.md`
2. `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
3. this file
4. the relevant bounded execution ledger only for the seam you choose

---

## Decision summary

If there is doubt what PromptManager should do next, the governing answer is:

**Strengthen the find -> understand -> decide -> reuse/refine loop around prompt assets, supported by compact trust surfaces, without drifting into a broader AI workstation.**
