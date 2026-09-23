# PromptManager — Status

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-09-23
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

## Current implementation checkpoint — 2026-09-23

**Active local slice — compact prompt-read JSON v2**
- `prompt-show`, `prompt-find`, and `prompt-history` now share one compact/full prompt-record contract for their JSON views: default JSON omits raw `ext4` and exposes bounded `embedding.present` / `embedding.dimensions` metadata.
- `--json --full` is the explicit complete-record opt-in and retains `ext4`; `--full` without `--json` is parser-invalid for all three commands.
- Scope is provider-free and read-only: prompt-history execution entries, semantic ranking, filters, persistence, embedding generation, and the stored `Prompt.to_record()` model remain unchanged.
- Release metadata advances to patch version `0.23.3`; verification results are recorded in the accompanying execution ledger before release closeout.

---

## Historical verified checkpoint — 2026-09-23

**Revision and delivery status**
- Current local `master` is based on `87f2a1e291122014ec2f17c414a5dcf2428f4921` before the uncommitted 0.23.1 release-closeout slice; `master...origin/master` is `0 / 0`.
- The working slice removes the remaining configured strict-Pyright debt without intended runtime, persistence, provider, or CLI-JSON contract changes. The local release candidate has `0 errors, 0 warnings, 0 informations` for full configured `pyright`.

**Current local verification**
- Full provider-free release gate: `892 passed, 1 skipped`, core coverage `81.68%`; `ruff check .`, `ruff format --check .`, CI-scope `pyright main.py config models`, full configured `pyright`, `uv lock --check`, and `git diff --check` all pass.
- The production cleanup keeps dynamic Qt/config/data boundaries typed locally with narrowed values, `cast`, typed callbacks, and small protocol adapters. Tests received only typing-stub cleanup; no user-facing behavior changed.
- GitHub CI still enforces the existing stable type subset. The full strict scan is verified locally for this release but is not described as CI parity until the workflow changes separately.

**Current release action**
1. Commit the verified strict-typing closeout and documentation sync.
2. Publish it as patch release `0.23.1` after updating package metadata, lockfile, and changelog.
3. Verify remote SHA parity and the exact-SHA Quality Gates result.

---

## Current verified checkpoint — 2026-09-22

**Revision and delivery**
- Current local `master`: `88ebff296ce084ea82d6d0e5be4c9259bedba42b`; the worktree was clean and `master...origin/master` was `0 / 0` immediately before this audit-closeout slice.
- Delivered after the prior `e218e37` checkpoint: catalog integrity checks, random prompt reads, prompt validation, deterministic template tests, prompt comparison, linting, effective template listing, tag operations, release `0.23.0`, and the Qt clipboard-test fix.
- This is a closed, provider-free supporting CLI set. It does not change the asset-first product center: each command organizes, inspects, validates, or compares persisted prompt assets.

**Current operational contracts**
- `catalog-check`, `prompt-validate`, `prompt-test`, `prompt-compare`, `prompt-lint`, and `prompt-template-list` are deterministic/provider-free by contract; `tag-list` and `tag-show` are read-only, and `prompt-tag` keeps an explicit non-mutating `--dry-run`.
- `catalog-check`, `prompt-show`, and `prompt-find` preserve LiteLLM availability state but suppress startup offline announcements because they remain useful local catalog operations without a configured model; `catalog-check --json` therefore emits a parseable JSON document. LLM-backed execution and generation paths retain offline guidance.
- `prompt-find <query>` sends the original natural-language query to raw semantic retrieval; explicit category/tag/source/active filters apply after ranking. Personalized, intent-hinted ranking remains in `suggest` and GUI recommendations.
- Text `prompt-show <uuid-or-name>` is a readable operator view with a blank-line-separated `<prompt_body>` / `</prompt_body>` copy block for non-empty context. Its default JSON form omits raw `ext4`; `--json --full` is the explicit complete-record opt-in.
- Root `--help` remains a grouped command card; use `<command> --help` for authoritative options.

**Current local verification**
- Full provider-free gate at the Linux/WSL GUI startup reliability checkpoint: `892 passed, 1 skipped`, core coverage `81.68%`; `ruff check .`, `ruff format --check .`, and CI-scope `pyright main.py config models` passed.
- Linux/WSL GUI startup now preflights PySide6's bundled `xcb` plugin before Qt can abort the process: unresolved system libraries produce controlled Ubuntu/Debian recovery guidance, without `sudo`, package installation, or host mutation. The duplicate `SettingsDialog` top-level layout warning is removed. Ledger: `docs/plans/2026-09-22-linux-wsl-gui-startup-reliability-v1.md`.
- `uv.lock` has been reconciled to package version `0.23.0`; `uv lock --check` passes after the lock-only update.
- The blocking CI type gate is intentionally `pyright main.py config models`, matching `.github/workflows/quality-gates.yml` and `docs/README-DEV.md`. Full configured strict Pyright remains a non-blocking debt scan; at this checkpoint it reports `283 errors`, chiefly outside the CI scope. It must not be described as green or as CI parity.
- The stale local `PROMPT_MANAGER_CONFIG_JSON` override remains an environment condition, not a repository regression: an explicit missing config path intentionally fails closed.

**Current next slice**
1. Active product sequence: `docs/plans/2026-09-22-instant-fit-judgment-v1.md` records the completed Stage A; `docs/plans/2026-09-22-intentional-capture-continuity-v1.md` is the active Stage B execution ledger.
2. Stage B adds only an explicit, operator-triggered Clipboard → editable Quick Capture draft preview; background watching, global hotkeys, automatic saves, providers, persistence changes, and a new capture flow remain out of scope.
3. Do not add another CLI, provider, chain, or integration surface without a separate product decision.
4. Treat Pyright expansion and Chroma/Dependabot review as separate, explicitly scoped maintenance work.

Historical checkpoints below retain their original revision-specific evidence.

---

## Historical verified checkpoint — 2026-09-20

The following ledger records the 2026-09-20 state and does not describe current `master`.

## Most recent delivered execution ledger at that checkpoint

Most recently delivered bounded execution ledger remembered in active planning docs:
- `docs/plans/2026-09-03-update-chroma-rollback-integrity.md` — synchronous update rollback.
- `a2a5d35` — async worker embedding rollback.

This is a status pointer only, not an instruction to continue that seam by default.

---

## Historical verified checkpoint — 2026-09-02

**Revision and delivery**
- Then-current local and remote `master`: `c2ebc73eb3f7764a799bdd1605b38f0f41af3750`.
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

## Latest bounded storage-integrity corrections — 2026-09-03

- Synchronous `update_prompt()` restores the prior SQLite prompt if Chroma embedding persistence fails, then re-raises the original storage error (`c93b892`).
- The async worker callback now likewise restores the prior SQLite `ext4` embedding when Chroma upsert fails, then re-raises so the existing worker retry can proceed (`a2a5d35`).
- Together, these repairs prevent SQLite from advancing while the derived semantic index retains the prior record; retry policy and the asynchronous no-embedding path are unchanged.
- Latest verification: `801 passed, 1 skipped`, coverage `80.38%`; exact-SHA Quality Gates succeeded for `a2a5d35`.
- Delivery ledger for the synchronous repair: `docs/plans/2026-09-03-update-chroma-rollback-integrity.md`.

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
