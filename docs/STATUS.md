# PromptManager — Status

Status: active
Owner: Wojtek / Prompt Manager Team
Updated: 2026-09-24
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

- GUI description-generation candidate, locally verified; remote delivery and exact-SHA CI to verify: the create/edit dialog gains an explicit Generate description action, while blank-description save and blank-name suggestions use deterministic local text only; name suggestion now waits until save, avoiding first-character names. A configured model request from Generate name/description asks before it can contact a provider; the CLI `prompt-add` description requirement remains unchanged. RED reproduction and isolated offline GUI creation flow are recorded in `docs/plans/2026-09-24-gui-description-generation-continuity-v1.md`. Full local gates passed (987 passed, 1 skipped, core coverage 81.66%, Ruff, strict Pyright, lock, diff check and wheel); independent replacement review found no blocker. Scope boundary: existing category insight/embedding persistence can still contact a configured provider; only implicit name/description generation was removed.
- Bounded CLI trust support: stage 1 `doctor [--json]` shipped as `7487bbf`; stages 2–5 shipped as `4e8c28fbb262ca00ec519c5ca5f8bef9de629a55` on `origin/master`, with exact-SHA Quality Gates [36004718272](https://github.com/voytas75/PromptManager/actions/runs/36004718272) successful. The latter adds `doctor catalog`, offline config/embedding readiness, local analytics counts, targeted prompt/chain checks and synced help/docs. Reads remain immutable and reports sanitized; CSV is created only with an explicit destination. The new opt-in `doctor embeddings --live` is locally verified: one approved synthetic Azure/LiteLLM request returned a usable 3072-dimensional vector (`EMBEDDING_BACKEND_REACHABLE`, exit 0, empty stderr). This proves backend response only, **not** vector-index health. Default `doctor` and `doctor embeddings` remain provider-free. Final local gates passed: 957 tests passed, 1 skipped, coverage 80.97%, Ruff, strict Pyright, uv lock and wheel build. The live slice was delivered as `355e6503b5ca1f618733dea9f77d063eb01cddc3`; exact-SHA Quality Gates [36008958726](https://github.com/voytas75/PromptManager/actions/runs/36008958726) passed. No second `doctor embeddings --live` probe was performed. The separate `doctor analytics --live` slice was delivered as `40fd155c70e62081f3de5b8f75135e5b391c2458` and [Quality Gates 36016455242](https://github.com/voytas75/PromptManager/actions/runs/36016455242) passed. It reuses the bounded backend check after a read-only count report; provider-free local verification passed (965 passed, 1 skipped; core coverage 81.41%, Ruff, strict Pyright, lock and wheel), and fake-provider entrypoint routing covers both paths. One separately approved installed-CLI live request returned exit 0, empty stderr, `EMBEDDING_BACKEND_REACHABLE` and dimension 3072; it did not inspect the vector index or establish analytics correctness. Default analytics stays provider-free. The on-demand `doctor index` metadata-only inspection is locally verified: immutable, sidecar-free readers compared exact catalog/Chroma metadata IDs without opening Chroma or contacting providers; actual installed-CLI read returned `INDEX_METADATA_MATCH`, 93 matching IDs, zero differences, `WARN`, and `vector_index=not_verified` (no HNSW/search claim). Full gates passed: 972 passed, 1 skipped, core coverage 81.66%, Ruff, strict Pyright, lock and wheel. Independent read-only review found no blockers; delivered as `63ca05d` plus docs checkpoint `35d3162`, with exact-SHA Quality Gates [36025044220](https://github.com/voytas75/PromptManager/actions/runs/36025044220) successful. Legacy commands remain available; alias retirement is a separate public-contract decision under `docs/plans/2026-09-24-doctor-cli-contract-and-migration-v1.md`.

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

## Current implementation checkpoint — 2026-09-24

**GUI/CLI tag parity — bounded correction**
- Promoted draft tags remain a comma-separated full-list replacement; CLI `prompt-tag` changes one tag. Both routes retain saved spelling. The GUI tag dropdown now uses the canonical CLI logical-tag catalog (case-insensitive, one option per tag), and selection/filter matching survive spelling changes.
- Focused regression covers case variants across prompts and refresh selection. No stored tag rewrite, provider call, or user database change. Verification and delivery state: `docs/plans/2026-09-24-gui-cli-tag-parity-v1.md`.

**Local-first GUI startup — bounded correction**
- Removed the default-mode gate that treated absent LiteLLM model/key as a blocker for opening the local GUI. Settings validation and local service initialization still own critical failures; diagnostics continue to report model availability, and execution stays guarded when offline.
- Added an isolated provider-free GUI operator-path regression: real entrypoint and Qt window, Quick Capture draft saved to temporary SQLite/ChromaDB, detail selection, Copy Prompt, Open in Workspace, and a refused model run.
- Verification: focused entrypoint/diagnostics tests and a real isolated offscreen GUI operator-path test; full provider-free gate `898 passed, 1 skipped` with core coverage `81.68%`; `ruff check .`, `ruff format --check .`, full configured strict `pyright`, `uv lock --check`, and `git diff --check` passed. See `docs/plans/2026-09-24-local-first-gui-offline-startup-v1.md` for the bounded ledger. No provider API, user database, or Windows checkout was exercised.

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

**Latest asset-loop slice — short search state continuity (delivered; exact-SHA CI verified)**
- A non-empty one-character toolbar search request now returns before changing active search, filter summary, or sort availability. Because no results were loaded, previously displayed results retain their own cues rather than being mislabeled as matches for the short text.
- Focused RED reproduced the stale-cue state; GREEN panel/controller suite: `14 passed`, and the complete suite: `958 passed, 1 skipped`, core coverage 80.97%, Ruff, format, strict Pyright, lock and diff check clean. Behavior commit `52f7cdd` and checkpoint `796f8bb` reached `origin/master`; exact-SHA Quality Gates [36012781306](https://github.com/voytas75/PromptManager/actions/runs/36012781306) succeeded. Ledger: `docs/plans/2026-09-24-short-search-state-continuity-v1.md`.

**Latest asset-loop slice — Recent reopen across narrowed search (delivered; exact-SHA CI verified)**
- The Recent dialog now reads catalog-wide recent prompts even when search/filter results are narrowed. A hidden selection preloads the catalog, clears active and pending narrowing, then shows the full list and selects the prompt in detail; a visible selection retains the current view. A failed read leaves the old view unchanged. No provider, ranking, or prompt-record mutation.
- Read-only reviews exposed three error/state blockers in successive candidates, each reproduced as RED and corrected. Final full local gate: 978 passed, 1 skipped, core coverage 81.66%, Ruff, strict Pyright, lock and diff check. Ledger: `docs/plans/2026-09-24-recent-reopen-search-continuity-v1.md`.
- Behavior commit `e8773fee4f9e51369aca9905d66b4726fea3e64d` reached `origin/master`; exact-SHA Quality Gates [36031535329](https://github.com/voytas75/PromptManager/actions/runs/36031535329) succeeded. Choose further asset-loop work only after fresh hesitation evidence.

**Older 2026-09-22 checkpoint (closed)**
1. `docs/plans/2026-09-22-instant-fit-judgment-v1.md` (Stage A) and `docs/plans/2026-09-22-intentional-capture-continuity-v1.md` (Stage B) are completed ledgers, not the active next item.
2. Stage B delivered the explicit Clipboard → editable Quick Capture draft preview. Background watching, global hotkeys, automatic saves, providers, persistence changes, and a new capture flow remain out of scope.
3. Another CLI, provider, chain, or integration surface needs a separate product decision.
4. Pyright expansion and Chroma/Dependabot review remain separately scoped maintenance work.

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
