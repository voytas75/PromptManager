# PromptManager — CLI asset-loop execution

Status: implementation closed; local acceptance passed. Commit/push authorized after this checkpoint; remote SHA and CI must be verified separately. Product authority: `docs/product-ssot.md`; near-term priority: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`. This is a bounded execution ledger, not a new product roadmap.

## Confirmed baseline and decision

At intake: clean `master` at `15ba0cb`, aligned with `origin/master`. Prior read-only audit: 42 root commands; full local suite 1070 passed, 1 skipped, core coverage 81.67%; Ruff, format, configured strict Pyright and lock check green. Explicit `pyright cli --outputjson` has 123 errors in `cli/commands.py` outside configured/CI scope. No provider call or live user-catalog write is authorized by this plan.

Work in priority order on the existing asset loop, not chain/evaluation/API expansion. After each slice record implementation, exact tests, scope, open doubts and next state here; do not mark pending work shipped. Use disposable catalogs and both installed and module CLI entrypoints. No dependency, security, CI, Windows checkout or broad SSOT changes. A local commit requires separately established authorization; push/publication/provider calls require a separate decision. Keep the worktree/diff honest at each checkpoint.

## Ordered slices

### P0 — prompt-add temporary payload lifecycle — implemented, verified locally

**Evidence:** `cli/parser.py:_write_temp_prompt_payload` creates `delete=False` JSON for inline/`--json` payload/`--from-stdin`; installed and module subprocess `--dry-run` each left two payload files in isolated `TMPDIR`. Sensitive prompt text persists even on preview.

**Scope:** `cli/parser.py`, `cli/entrypoint.py`, `main.py`, focused tests (prefer `tests/test_main_entry.py` or a dedicated process test), `docs/CHANGELOG.md` and this ledger. First RED on both entrypoints for preview, apply and failure after parsing; also cover caller-supplied file untouched and no duplicate parse/materialization. Then minimal cleanup in `finally` across startup/handler failures without changing import result or exit/output contracts. Probe parsing-error path separately: do not imply cleanup before a temp path exists. Provider-free and isolated; inspect actual bytes/absence after process exit.

**Verification:** targeted tests, real subprocess smoke for both entrypoints, Ruff check/format, configured Pyright plus targeted Pyright on changed code where meaningful, full provider-free pytest + `--cov-fail-under=80`, `uv lock --check`, `git diff --check`, status/numstat. Record exact outcomes here before transition.

**Implemented:** the initial parser marks only its generated payloads; both installed and module entrypoints pass the already parsed namespace to the application, which unlinks generated files on dispatch/startup exit. The installed entrypoint also covers failure importing the application. Caller-supplied file paths are not deleted. RED reproduced leftover files in `tests/test_prompt_add_temp_cleanup.py`; GREEN matrix exercises inline, JSON, stdin and caller file × preview, apply and missing-config startup × both process entrypoints (24 cases). No change to the importer or provider/CI configuration.

**Independent review and repair:** read-only reviewer reproduced two previously uncovered failures: the `python -m main` early parse could leave its payload when a later module import fails, and partial JSON writing could leave a file before `args` gained its cleanup marker. Added two startup-import guards in `main.py` (preserving the doctor-first path) and a write-failure guard in `cli/parser.py`. RED tests reproduced both failures; GREEN tests: 3 passed (installed/module startup import, interrupted JSON write). The installed entrypoint already covered its import failure. No claim for a hard process kill.

**Verified at the pre-delivery local checkpoint:** post-review `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `.venv/bin/pyright`, `uv lock --check` and `git diff --check`: pass. `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q -n auto --cov=core --cov-report=term --cov-fail-under=80`: 1106 passed, 1 skipped; core 81.67%. Explicit Pyright on `main.py cli/parser.py tests/test_prompt_add_temp_cleanup.py`: 0 errors. Subprocess matrix uses disposable HOME, TMPDIR, DB, config, deterministic embeddings and empty model settings; startup-import/write-failure tests use isolated temporary directories and controlled faults. No provider or live-catalog write was required. This paragraph records the local checkpoint, not the remote delivery state.

### P1 — draft promotion CLI — closed by owner decision: keep promotion in GUI

**Decision evidence at intake (superseded by the owner decision below):** `draft` exposes add/show/find/delete but no promotion; GUI has `promote_draft_prompt`; `prompt-add` can overwrite an existing draft by name and clear draft state while retaining ID, without UUID or optimistic-concurrency intent. Do not call that path safe promotion.

**Historical decision gate (closed by owner decision below):** inspect GUI/model/repository update lifecycle, duplicate-name semantics, cache and vector indexing, provider egress, and draft version/activity invariants before any future CLI promotion. A UUID-based preview/apply with an expected modification token was a candidate, not approved work. Do not improvise SQL to bypass domain lifecycle.

**Decision evidence:** GUI `PromptEditorFlow.promote_draft_prompt` builds a modified prompt via `build_promoted_prompt` then invokes `manager.update_prompt`. That lifecycle performs embedding provider work or schedules an embedding worker, updates SQLite/version/activity and may affect Chroma/cache. The existing provider-free `cli/draft.py` instead directly opens the selected SQLite catalog and intentionally does not initialize that manager. Its delete path fails closed with Redis configured and carefully inspects an existing Chroma vector. `PromptRepository.update`/`update_with_version` have no expected-modified compare-and-set; `PromptRepository.update_with_version` does transactionally couple record and version but no cache/index/provider guarantees. Thus there is **no verified small existing provider-free promotion seam** satisfying the proposed optimistic concurrency and index/cache invariants. SQL-only promotion would silently change semantics and risks a stale vector/cache; full manager route could contact a provider. No promotion code changed.

**Owner decision:** keep draft promotion in GUI for now; close this campaign after verified CLI fixes. This is a deliberate product boundary, not an unimplemented CLI promise. Reopen only if a real non-GUI promotion need is demonstrated; then choose explicit offline fail-closed or provider-capable semantics separately, with concurrency/index/cache contract and authorization for any paid live acceptance. `prompt-add` by name is not a supported substitute.

### P2 — machine-readable find/show errors — implemented, verified locally

**Evidence:** isolated `prompt-show <missing> --json` returned text stdout plus stderr log; `prompt-find <none> --json` returned text instead of JSON. `prompt-add --json` means input, not output. First define backward-compatible scope and run a real subprocess matrix for success, empty, missing, invalid and startup errors in both entrypoints. Repair only the selected `find → show` paths with additive output conventions and focused tests; keep human output and existing `prompt-add` input flag unchanged. Do not claim whole CLI agent-ready or silently alter exit codes.

**Implemented:** RED for JSON-mode missing/ambiguous show, empty/invalid find; only selected command-handler error branches changed. Empty `prompt-find --json` emits `[]` with exit 0; missing/ambiguous/lookup failure for show and invalid query/active or search failure for find emit sanitized `{ok:false,error:{code,message}}` on stderr with stdout empty, retaining existing return codes. Existing text paths and successful JSON record/array shapes remain unchanged. Docs updated in `docs/README-DEV.md` and changelog. No parser/other JSON command family, ranking, provider or model changes.

**Verified at the earlier P2 local checkpoint:** real subprocess `tests/test_prompt_read_json_cli.py` exercises installed/module entrypoints against a disposable catalog: 2 passed, including successful show, no matches, missing and invalid filter; focused P0+P2 subprocess pack 26 passed. Handler tests also cover ambiguous names, lookup/search failures and whitespace query without leaking backend detail. Full `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q -n auto --cov=core --cov-report=term --cov-fail-under=80`: 1103 passed, 1 skipped, core 81.67%. Ruff check/format, configured strict Pyright, lock check and diff whitespace check pass. Explicit Pyright on `cli/commands.py` still reports 123 errors, the inherited excluded-file baseline; no new diagnostic observed in changed `find/show` lines. See remaining P3 decision. No commit or push had been made at this checkpoint.

### P3 — remaining improvement decision — closed: no further code slice justified

After P0–P2, measure operator friction on draft-to-asset and find→inspect→reuse. Consider only one bounded CLI typing/contract follow-up if it blocks those paths. Otherwise close the campaign. Defer dependency/impact graphs, provider evaluation, chain expansion, dashboards, HTTP API and repo-wide Pyright/CI changes absent a separate need and authorization.

**Decision and evidence:** two concrete CLI defects P0/P2 have test-backed fixes. No demonstrated further operator blocker suitable for a third code change; draft promotion remains GUI-only by owner decision. An explicit `pyright cli/commands.py --outputjson` after the edits still has 123 inherited excluded-file diagnostics and none in the changed `find/show` lines; configured Pyright is green. Do not open a broad typing campaign by inertia. This campaign's implementation work is locally complete.

**Pre-delivery scope checkpoint:** changed tracked files: `cli/commands.py`, `cli/entrypoint.py`, `cli/parser.py`, `main.py`, `tests/test_main_entry.py`, `docs/CHANGELOG.md`, `docs/README-DEV.md`. New files: this execution ledger, `tests/test_prompt_add_temp_cleanup.py`, `tests/test_prompt_read_json_cli.py`. Inspect the staged numstat (including new files) before committing. At this checkpoint no commit SHA, remote delivery, exact-SHA CI, Windows acceptance or provider-backed test existed for these changes.

## Current blockers and doubts

- P0: resolved for normal and missing-config startup by passing the first parse forward. A hard process kill cannot guarantee cleanup of any temporary file; this is a residual OS/process-lifecycle limitation, not a covered success path.
- P1: GUI-only promotion is the chosen scope. If a repeatable non-GUI need appears, verify the offline/provider, concurrency and index/cache contract before reopening; none is implied by this campaign.
- P2: consumers of legacy text/error output are unknown; text mode preserved. JSON success shapes and exit codes retained; only selected JSON error/empty surfaces improved. Global startup or parser errors are not part of this bounded JSON guarantee.

## Closure / reopening trigger

P0 and P2 passed local acceptance; commit and push were subsequently authorized. Verify the three remote/local SHAs and exact-SHA CI after publishing; this ledger records implementation and the pre-delivery checkpoint, not a future CI conclusion. Draft promotion stays in GUI; do not use `prompt-add` by name as a substitute. Reopen development only for a demonstrated non-GUI promotion need or independently reproduced CLI defect. Native Windows acceptance remains separate and unverified.
