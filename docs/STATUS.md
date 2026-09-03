# PromptManager — Status

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-09-03
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
- `docs/plans/2026-09-03-update-chroma-rollback-integrity.md`

This is a status pointer only, not an instruction to continue that seam by default.

---

## Current verified checkpoint — 2026-09-02

**Revision and delivery**
- Current local and remote `master`: `c2ebc73eb3f7764a799bdd1605b38f0f41af3750`.
- Worktree was clean and `HEAD...origin/master` was `0 / 0` before this status update.
- Recent delivered commits:
  - `522edd4` — `fix(config): persist settings to active config path`.
  - `c2ebc73` — `fix(storage): preserve prompt when Chroma delete fails`.
- Exact-SHA Quality Gates for `c2ebc73` succeeded: Ruff, formatter verification, CI-scope Pyright, pytest, and clean-tree check.

**Delivered contract repairs**
- GUI settings persistence now writes to the active `PROMPT_MANAGER_CONFIG_JSON` path; default behavior remains `config/config.json`.
- `delete_prompt()` deletes the Chroma record before SQLite. A Chroma deletion error therefore preserves the SQLite prompt record.
- Local verification for the two repairs: `795 passed, 1 skipped`; Ruff, formatter check, CI-scope Pyright, and `git diff --check` passed before delivery.

**Security checkpoint — ChromaDB**
- REST and GraphQL report four open Dependabot alerts, all for direct `chromadb==1.5.7` in `uv.lock`: `#65`, `#99`, `#100`, and `#101` (two critical, two high).
- The vulnerable ranges extend through `chromadb==1.5.9`; Dependabot reports no patched version. A scratch-only `1.5.7 -> 1.5.9` resolution changed only `uv.lock` (`7 additions / 7 deletions`) and would not close these alerts.
- The accepted-but-open decision in `docs/plans/2026-09-02-dependabot-remediation-campaign.md` remains authoritative: ChromaDB is limited to local `PersistentClient` / `EphemeralClient` use. HTTP/server, non-local, remote, and multi-tenant use are prohibited pending a new decision or upstream patch.
- A later full scratch verification attempt did not start because its temporary worktree command failed with `fatal: not a git repository`; it is not validation evidence and must be recreated only if an upstream patch creates a viable remediation candidate.

**Resume rule — one next slice only**
1. Do not retry a ChromaDB upgrade until an upstream version above `1.5.9` is published or a non-local/server Chroma proposal appears.
2. When that trigger occurs: first create a fresh disposable worktree from current `master`, resolve the smallest ChromaDB candidate there, measure its exact diff, run provider-free storage/telemetry tests, then request separate approval before modifying the primary worktree.
3. Without that trigger, return to the product SSOT priority: read-only probe one current retrieval → inspect → reuse hesitation seam; do not reopen delivered wording slices without a reproduced gap.

---

## Latest bounded product correction — 2026-09-03

- Draft prompts found by title no longer receive the list-side `Ready to reuse` handoff.
- The retrieval reason remains `Matched in title`; `Promote Draft` remains the canonical detail-side action.
- Verified locally: 23 focused list tests; 73 capture/list/detail parity tests; active-path Pyright, Ruff, and format checks passed.
- Delivery ledger: `docs/plans/2026-09-03-draft-title-match-handoff-consistency-roadmap.md`.

---

## Latest bounded storage-integrity correction — 2026-09-03

- `update_prompt()` now restores the prior SQLite prompt if synchronous Chroma embedding persistence fails, then re-raises the original storage error.
- This prevents SQLite from advancing while the derived semantic index retains the prior record; the asynchronous no-embedding path is unchanged.
- Verified locally: 2 focused failure-path tests; 55 storage/branch tests; active-path Pyright, Ruff, and format checks passed.
- Delivery ledger: `docs/plans/2026-09-03-update-chroma-rollback-integrity.md`.

---

## Latest bounded storage-integrity correction — 2026-09-03

- `update_prompt()` now restores the prior SQLite prompt if synchronous Chroma embedding persistence fails, then re-raises the original storage error.
- This prevents SQLite from advancing while the derived semantic index retains the prior record; the asynchronous no-embedding path is unchanged.
- Verified locally: 2 focused failure-path tests; 55 storage/branch tests; active-path Pyright, Ruff, and format checks passed.
- Delivery ledger: `docs/plans/2026-09-03-update-chroma-rollback-integrity.md`.

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
