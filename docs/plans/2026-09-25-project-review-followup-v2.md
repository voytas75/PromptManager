# PromptManager — przegląd projektu: działania uzupełniające (v2)

Status: steps 1–3 verified locally; delivery state is determined by Git and exact-SHA CI
Date: 2026-09-25
Starting revision: `55e20cf7ba0874e93488c8f0a332def6092a53ec` (`master`, clean and aligned with `origin/master` at intake)
Product authority: `docs/product-ssot.md`; near-term priorities: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`; delivered history: `docs/STATUS.md`.

## Scope and evidence

This is a bounded follow-up ledger for the 2026-09-25 review, **not** a replacement product roadmap. Fresh read-only review: 998 passed, 1 skipped; core coverage 81.66%; Ruff, format, lock, configured strict Pyright (251 files, 0 errors), and exact-SHA Quality Gates for `55e20cf` green. Explicit strict `pyright cli --outputjson` reports 127 errors, all in `cli/commands.py` outside configured and CI scopes. Focused offline GUI/chain pack: 97 passed. At intake, the old review-actions ledger still said uncommitted although its three changes had already been delivered at `55e20cf` and CI run `36130725473` had succeeded.

Boundaries for the work itself: no provider calls, user-catalog writes, dependency/CI/security-policy changes, broad refactor or feature removal. Do not infer a chain bug from module coverage alone. Commit and push, when separately authorized, require Git/remote/exact-SHA CI readback; local gates alone do not prove delivery. After **each** step record implementation/decision, exact verification, and next active step. On a failed gate, mark blocked rather than complete.

## Ordered actions

### 1. Correct the delivered review ledger — verified

Target: `docs/plans/2026-09-25-project-review-actions-v1.md`. Replace stale *current* local/uncommitted status and closeout with exact shipped SHA and exact-SHA CI evidence. Preserve the historical fact that the changes were uncommitted at each earlier checkpoint; do not rewrite those historical notes as if delivery had already occurred.

Done when: active status and closeout match Git/CI; historical checkpoints remain truthful; `git diff --check` passes.

Implemented: replaced the v1 ledger's current local/uncommitted status and closeout with the delivered commit `55e20cf7ba0874e93488c8f0a332def6092a53ec`, successful exact-SHA CI link, and this follow-up pointer; retained the earlier pre-commit notes as historical checkpoints.
Verified: `git` at intake showed clean `master...origin/master` and zero divergence at `55e20cf`; `gh run list --commit 55e20cf7ba0874e93488c8f0a332def6092a53ec` reported run `36130725473` completed successfully; targeted ledger assertions and `git diff --check` passed. Local follow-up changes remain uncommitted; no new remote claim. Next: step 2.

### 2. Reduce one verified CLI typing cluster without changing behavior — verified

Candidate: `run_benchmark` text-output `usage_parts` accumulator in `cli/commands.py`; Pyright flags its append/join operations. First check corresponding provider-free fake-manager tests and before-change diagnostics. If the candidate cannot be tested without a provider, stop and record a new recommendation rather than calling one. Use a single `list[str]` annotation only if it removes the targeted errors without changing output. Run focused tests before/after, explicit CLI Pyright (remaining baseline), Ruff/format, configured Pyright, full provider-free pytest with core coverage and lock check.

Done when: targeted diagnostics disappear; benchmark text-output test passes before/after; total CLI errors decrease from 127; configured gates remain green. Remaining errors are debt, not an all-CLI PASS.

Implemented: added a provider-free fake-manager regression in `tests/test_cli_benchmark_output.py` for benchmark token/preview text, then annotated only `run_benchmark`'s `usage_parts` as `list[str]` in `cli/commands.py`; no output, provider, persistence or CI change.
Verified: before change new focused test passed (1); Pyright reported 127 CLI errors, four in the target lines 372/374/376/377. After change focused benchmark/manager pack: 11 passed; targeted errors 0, CLI total 123. Test file Pyright 0. Whole-repo Ruff and format passed (266 files formatted); configured Pyright 252 files/0 errors; `uv lock --check` and `git diff --check` passed. Full provider-free suite 999 passed, 1 skipped; core coverage 81.66%. Local only; the 123 remaining CLI errors are not a pass or CI scope. Next: step 3.

### 3. Decide whether chain coverage warrants a targeted next slice — verified (decision only)

Inspect chain runtime/repository low-covered lines and nearest tests against an actual user-visible/failure contract. Do not implement coverage-driven expansion by default. If no reproduced high-impact gap exists, park until a chain behavior change or concrete failure; keep asset-loop retrieval/inspect/reuse priorities ahead of new chain features.

Done when: evidence and a reopen trigger are recorded, with no unproven unused-code claims.

Implemented/decision: no chain implementation. The low coverage (36% for `core/prompt_manager/chains.py`, 17% for `core/repository/chains.py`) is a measurement, not evidence of dead code or a specific defect. `ChainStoreMixin` has concrete list/get/add/update/delete/existence persistence operations (`core/repository/chains.py:57–168`), consumed by the manager (`core/prompt_manager/chains.py:168–205`); the runner processes ordered steps and records terminal status (`core/prompt_manager/chains.py:208–380`). Chain model tests explicitly preserve legacy fields as inactive compatibility metadata; do not remove them or enlarge chain scope merely because of line coverage. Defer additional chain work in favor of the asset-loop priority.
Verified: targeted chain backend/model/CLI/dialog pack: 96 passed; full provider-free suite after the code slice: 999 passed, 1 skipped, core coverage 81.66%. No focused chain failure was reproduced. Reopen on an isolated failing chain import/persistence/run contract, a user-visible chain-result ambiguity impacting reuse/refine, or an approved chain behavior change; then add the nearest focused RED test before altering runtime. Chain usage in production remains **do weryfikacji**; none was measured here.

## Recommendation and counterfactual

Prefer a reproduced operator obstacle in find → inspect → reuse/refine over continued technical cleanup if an isolated test or observed operator path demonstrates a higher-impact defect. Otherwise finish only the bounded trust slice above. Any successor must be selected from the near-term plan with fresh code/test evidence, not automatically inherited from this ledger.

## Closure and next decision

All three bounded review follow-ups have local verification. The old delivery status is corrected; the benchmark CLI text contract now has a provider-free guard, and four targeted Pyright errors are gone (127 → 123); chains remain parked without a reproduced contract failure. The next *candidate*, not an automatic continuation of this ledger, is a concrete retrieval → inspect → reuse/refine operator obstacle selected from the active near-term plan after a fresh test/runtime check. Another CLI typing slice is a supporting alternative only if its diagnostic blocks that path. Keep the remaining 123 CLI errors outside CI claims. Delivery is a separate Git/remote/CI checkpoint; no SHA or CI for **this** follow-up is asserted by the local tests above.
