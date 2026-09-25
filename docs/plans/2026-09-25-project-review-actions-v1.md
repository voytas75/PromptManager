# PromptManager — działania po przeglądzie projektu (v1)

Status: completed locally; no commit or push
Date: 2026-09-25
Product authority: `docs/product-ssot.md`
Near-term priority authority: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`
Scope: bounded follow-through on the 2026-09-25 read-only project review, not a new product roadmap.

## Baseline and boundaries

- Starting revision: `c0df5b0`, clean `master...origin/master` at intake.
- Local gates at review: 998 passed, 1 skipped; `core` coverage 81.66%; Ruff, `uv lock --check`, configured strict Pyright (251 checked files, zero errors) green; exact-SHA Quality Gates successful for `c0df5b0`.
- Explicit `pyright cli --stats` found 129 errors, all in `cli/commands.py`. The configured full scan excludes `cli/`; CI checks only `main.py config models`. Do not describe either as all-project strict typing.
- No provider calls, user catalog mutations, dependency changes, CI policy changes, broad refactor or automatic deletion. No changes to `docs/product-ssot.md` without changed product truth. Commit and push are outside this run's scope.
- Update this ledger after **each** completed step with `Implemented`, exact `Verified` evidence, and the next active step. If a gate fails, mark the step blocked rather than completed.

## Ordered steps

### 1. Reconcile stale execution-plan pointer — completed

Target: `docs/plans/2026-04-25-roadmap-implementation-plan.md`, plus this ledger. The historical roadmap calls the finished `detail-edit-vs-fork` slice active, although its own tasks are complete and the current near-term plan is elsewhere. Replace only the stale current-pointer wording, preserving historical delivery references and the authority hierarchy. Do not make this review ledger a competing strategic plan.

Done when: the old plan points to product SSOT, active near-term plan and `docs/STATUS.md`; the delivered detail slice is not called active; links/claims and `git diff --check` pass.

Implemented: demoted the April execution plan to historical context, replaced its false active-successor pointer with the current product/near-term/status hierarchy, and retained the delivered ledger links. Product SSOT unchanged.
Verified: checked the referenced paths (10 references in inspected control sections; none missing); checked the target detail ledger reports `delivered`; stale `next active bounded execution ledger` pointer absent; `git diff --check` passed. Local uncommitted documentation checkpoint, no remote claim.

### 2. One behavior-neutral CLI typing cluster — completed

Target: the `run_prompt_find` text-output `lines` accumulator in `cli/commands.py` (two strict-Pyright errors at append/join), guarded by existing focused `tests/test_main_entry.py` prompt-find text/JSON/filter tests. The fresh machine-readable `pyright cli --outputjson` scan found 129 errors in `commands.py`, concentrated in usage/benchmark, chain/compare and analytics blocks; this two-error asset-retrieval seam is the smallest behavior-owned slice. Preserve CLI text/JSON and error semantics; do not include all of `cli/` in configured/CI blocking Pyright yet.

Done when: the chosen cluster has no Pyright errors, focused behavior tests and Ruff pass, the entire `cli/` count is remeasured and lower than 129, and full configured Pyright plus provider-free pytest/coverage remain green. Record any remaining errors as debt, not as a claimed clean CLI.

Implemented: annotated the existing `run_prompt_find` text-line accumulator as `list[str]` in `cli/commands.py`. No output branch, filtering, ranking, JSON schema, error handling, dependency or CI scope changed; no new test was required because existing text/JSON/filter tests exercise the seam.
Verified: before change, 9 focused prompt-find tests passed; after change, the same 9 passed (103 deselected). Machine-readable strict Pyright for `cli/` fell from 129 to 127 errors; no targeted append/join errors remain. `ruff check` and `ruff format --check` on the file and whole repo passed; configured Pyright checked 251 files with 0 errors; `uv lock --check`, `git diff --check` and full provider-free suite passed (998 passed, 1 skipped; core coverage 81.66%). The remaining 127 CLI errors are **not** covered by configured/CI Pyright. No commit or push.

### 3. Complexity/feature scope checkpoint — completed (decision only)

Evidence-only decision after step 2. Inspect the active call sites for chains/sharing/analytics and the owner seams of the large manager/detail/CLI modules. Record the stop rule: do not delete active features or mechanically split large files merely due to size; revisit only after a reproduced operator friction or change-locality failure. Do not initiate architecture extraction, dependency changes, or a new roadmap under this step.

Done when: this ledger records the decision, supporting code references and a concrete reopening trigger; the active near-term plan remains product authority. If there is no evidenced change to make, close as `no implementation` rather than manufacturing a slice.

Implemented: no implementation or feature removal. Keep chains, sharing and analytics subordinate and unchanged: GUI and CLI invoke `run_prompt_chain` (`gui/dialogs/prompt_chains.py:1621`, `cli/commands.py:1183`), the GUI constructs `ShareWorkflowCoordinator` and calls its `share_prompt` path (`gui/main_window_bootstrapper.py:157`, `gui/main_window_handlers.py:360–365`), and the Analytics tab builds a snapshot (`gui/main_view_builder.py:449–460`, `gui/analytics_panel.py:189`). The large manager composition (`core/prompt_manager/__init__.py:201–219`), shared detail widget and CLI handler file are change-locality watchpoints, not proven dead architecture.
Verified: static call-site search and focused file reads; `docs/product-ssot.md:403–411,415–486` keeps supporting layers below the asset loop, and `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` remains near-term authority. Reopen an extraction only on a reproduced multi-file change-locality failure or failing contract test attributable to that boundary; reopen a feature cut only on observed operator friction/usage evidence, not line counts. No provider or user data accessed.

## Alternative and decision signal

Strongest alternative: prioritize a newly reproduced find → inspect → reuse obstacle over typing work. Switch order only if an isolated user-flow test or observed operator session demonstrates a concrete, higher-impact obstacle. Otherwise complete the bounded trust/debt slice first.

## Closure and next decision

All three review actions are completed **locally**. The historical pointer is corrected, the smallest retrieval-owned CLI type cluster is green while 127 CLI errors remain outside the configured gate, and feature/complexity cuts are deliberately deferred for lack of a reproduced need. No new slice is authorized by this ledger. Choose any successor from the active near-term plan against fresh operator evidence; treat broader CLI typing, CI expansion, dependency changes, releases and user-data repair as separate scope decisions. The three changed files remain uncommitted; no remote or exact-SHA CI claim for these edits.
