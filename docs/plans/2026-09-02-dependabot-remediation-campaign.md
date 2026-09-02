# PromptManager — Dependabot remediation campaign

Status: active
Owner: Wojtek / Prompt Manager Team
Created: 2026-09-02
Canonical product SSOT: `docs/product-ssot.md`

## Goal

Reduce confirmed Dependabot exposure in bounded, independently verifiable dependency slices without weakening the local-first, asset-first product boundary or treating a green test run as alert closure.

## Baseline

Confirmed on 2026-09-02:

- REST and GraphQL initially reported 38 open Dependabot alerts; after Phase 2 remote delivery, both report 22.
- The alerts reduce to seven package families: `litellm`, `aiohttp`, `cryptography`, `vcrpy`, `pydantic-settings`, `idna`, and `chromadb`.
- `chromadb` has four open alerts without a published patched version and is currently used through local `PersistentClient` / `EphemeralClient` code paths, not a configured HTTP/server client.
- The current default-branch Quality Gates run for `f598d2e` is green.
- `uv lock --check` and `uv pip check --python .venv/bin/python` pass before remediation.

## Campaign order

### Phase 1 — LiteLLM / AIOHTTP resolution probe

**Goal:** determine the smallest resolver-valid patch set that raises LiteLLM to at least `1.84.0` and resolves the AIOHTTP advisory floor of `3.14.3`.

**Boundary:** use a disposable worktree only. Do not modify the primary worktree, commit, or push during the probe.

**Entry condition:** clean primary worktree and current `master` baseline.

**Required evidence:**
- exact `pyproject.toml` / `uv.lock` diff and gross line count,
- resolved versions for `litellm` and `aiohttp`,
- `uv lock --check`, frozen sync, `uv pip check`, and targeted runtime/quality probes,
- independent same-interpreter audit where feasible.

**Exit condition:** either a verified candidate diff is ready for explicit scope approval, or an exact resolver/runtime blocker is recorded.

**Phase 1 result — 2026-09-02:**
- Candidate resolution: `litellm==1.84.0` and transitive `aiohttp==3.14.3`.
- Candidate scope: `pyproject.toml` and `uv.lock`; 72 additions / 59 deletions, **131 gross lines**.
- Candidate verification in a disposable worktree: `uv lock --check`, frozen `uv sync --extra dev`, `uv pip check`, full pytest (`794 passed, 1 skipped`, coverage `80.01%`), Ruff, and CI-scope Pyright all passed.
- A first targeted test run failed only because the isolated checkout lacked ignored `config/config.json`; the rerun with the CI-equivalent transient `{}` config passed (`78 passed`). The config file is not part of the candidate diff.
- Same-interpreter `pip-audit` still reports other package families (`chromadb`, `cryptography`, `idna`, `pydantic-settings`, `vcrpy`) and an advisory for transitive `click==8.1.8`; these are deferred and do not expand this candidate batch.

### Phase 2 — Approved LiteLLM / AIOHTTP remediation

**Goal:** apply only the Phase 1-approved package/lock changes in the primary worktree.

**Entry condition:** explicit approval of the measured file and gross-line scope.

**Verification:** focused tests for execution/settings plus repository gates appropriate to the changed dependency surface; after push, exact-SHA CI, Dependency Graph, and REST + GraphQL alert recheck.

**Local result — 2026-09-02:**
- Applied the approved `pyproject.toml` / `uv.lock` batch: `litellm==1.84.0`, transitive `aiohttp==3.14.3`.
- Re-measured primary-worktree scope exactly as approved: 72 additions / 59 deletions, **131 gross lines**.
- Primary-worktree verification passed: frozen sync, `uv pip check`, full pytest (`794 passed, 1 skipped`, coverage `80.39%`), Ruff, formatter check, and CI-scope Pyright (`0 errors`).
- Pending separately authorized commit/push: GitHub CI, Dependency Graph, and alert closure cannot yet be claimed.
- Remote result after commit `7b32425`: exact-SHA Quality Gates and Dependency Graph both succeeded; REST and GraphQL open-alert counts fell from 38 to 22.

### Phase 3 — Cryptography remediation

**Goal:** independently move direct `cryptography` to at least `50.0.0` and validate PrivateBin encryption/share behavior.

**Boundary:** separate commit from Phase 2.

**Verification:** scratch resolution first; then focused sharing tests and full dependency/gate verification after approved scope.

**Phase 3 scratch result — 2026-09-02:**
- Candidate resolution: direct `cryptography==50.0.0`.
- Candidate scope: `pyproject.toml` and `uv.lock`; 44 additions / 47 deletions, **91 gross lines**.
- No unrelated resolver movement observed: `litellm==1.84.0`, `aiohttp==3.14.3`, `cffi==2.0.0` remain stable.
- Scratch verification passed: frozen sync, `uv pip check`, `tests/test_sharing.py` (`9 passed`), targeted Ruff, targeted Pyright, and runtime version probe.
- Candidate is ready for explicit scope approval; no primary-worktree changes were made.

**Primary-worktree result — 2026-09-02:**
- Applied the approved `pyproject.toml` / `uv.lock` candidate with the measured 91-gross-line scope.
- Frozen sync, `uv pip check`, `tests/test_sharing.py` (`9 passed`), targeted Ruff, targeted Pyright, and runtime `cryptography==50.0.0` verification passed.
- Pending separately authorized commit/push: remote CI, Dependency Graph, and Dependabot closure are not yet claimed.
- Remote result after commit `3f01e30`: exact-SHA Quality Gates and Dependency Graph both succeeded; REST and GraphQL open-alert counts fell from 22 to 18.

### Phase 4 — Dev and settings dependencies

**Goal:** resolve direct `vcrpy` to at least `8.2.1` and `pydantic-settings` to at least `2.14.2`; include `idna` in a fresh current-master lock resolution rather than merging stale Dependabot PR #16.

**Boundary:** split into independently measured batches if the resolver creates unrelated churn.

**Verification:** targeted test collection/import coverage, frozen lock, package check, full or risk-proportionate quality gates, then GitHub alert recheck.

**Phase 4 scratch result — 2026-09-02:**
- Candidate resolution: `pydantic-settings==2.14.2`, `vcrpy==8.2.1`, and transitive `idna==3.19` from a fresh current-master lock resolution.
- Candidate scope: `pyproject.toml` and `uv.lock`; 13 additions / 13 deletions, **26 gross lines**.
- This supersedes the stale Dependabot PR #16's lock-only `idna==3.15` proposal; no merge of that stale branch is needed.
- No unrelated resolver movement observed: `cryptography==50.0.0`, `litellm==1.84.0`, `aiohttp==3.14.3`, and `click==8.1.8` remain stable.
- Scratch verification passed: frozen sync, `uv pip check`, focused settings/main-entry/sharing tests (`116 passed`), targeted Ruff, and targeted Pyright.
- Candidate is ready for explicit scope approval; no primary-worktree dependency changes were made.

**Primary-worktree result — 2026-09-02:**
- Applied the approved `pyproject.toml` / `uv.lock` candidate with the measured 26-gross-line scope.
- Frozen sync, `uv lock --check`, `uv pip check`, focused settings/main-entry/sharing tests (`116 passed`), targeted Ruff, targeted Pyright, and runtime version probes passed.
- Pending separately authorized commit/push: remote CI, Dependency Graph, and Dependabot closure are not yet claimed.
- Remote result after commit `03e37a6`: exact-SHA Quality Gates and Dependency Graph both succeeded; REST and GraphQL open-alert counts fell from 18 to 15.

### Phase 5 — ChromaDB risk decision

**Goal:** record an explicit temporary decision for the four unpatched ChromaDB advisories.

**Evidence — 2026-09-02:**
- Current resolved package: `chromadb==1.5.7`; Dependabot lists no patched release for any of the four alerts, whose vulnerable ranges extend through `1.5.9`.
- Affected alerts remain open in `uv.lock`: `GHSA-f4j7-r4q5-qw2c` and `GHSA-36p7-vc44-83pf` (critical code injection), plus `GHSA-xph7-9rjv-w5fr` and `GHSA-2wm9-hf6c-p5cr` (high tenant/RBAC authorization flaws).
- The executable client construction uses only `chromadb.PersistentClient` and fallback `chromadb.EphemeralClient` in `core/prompt_manager/backends.py`; static search found no `HttpClient`, Chroma host configuration, or server endpoint wiring in application code.

**Temporary risk decision — accepted but open:**
- PromptManager may use ChromaDB only through its local persistent or ephemeral client paths.
- PromptManager must not expose ChromaDB through HTTP, run a Chroma server, or use the package for remote multi-tenant storage under this decision.
- The four ChromaDB Dependabot alerts remain open and are not represented as fixed.
- This acceptance applies only to the server/RBAC attack surface examined here; it does not authorize unrelated ChromaDB features or deployments.

**Re-evaluation trigger:**
- an upstream patched release becomes available, or
- any proposal introduces Chroma HTTP/server mode, non-local storage, remote access, or multi-tenant use.

**Exit condition:** the risk is documented as accepted-but-open, never presented as fixed.

### Post-remediation Dependabot reconciliation — 2026-09-02

**Verified external closeout:**
- Nine stale alerts were dismissed as `inaccurate` after their current default-branch pins and successful Dependency Graph runs were checked: LiteLLM `#78`, `#85`, `#86`, `#94`, `#95`; Cryptography `#77`, `#96`, `#97`; and alert `#98` was withdrawn by GitHub.
- VCR.py `#81` and Pydantic Settings `#82` were then dismissed as `inaccurate` after the token received `admin:repo_hook`; their default-branch pins equal the respective patched floors.
- REST and GraphQL now agree on **4 open alerts**, all accepted-but-open ChromaDB advisories: `#65`, `#99`, `#100`, and `#101`.

**Campaign closeout:**
- No remediation is pending for patched dependencies.
- Do not open another dependency-upgrade slice for alerts that already have safe pins.
- Monitor for a ChromaDB upstream patch, a new alert for the current resolved graph, or an architectural proposal for Chroma HTTP/server, non-local storage, remote access, or multi-tenancy.

## Cross-phase rules

1. A green local gate is not Dependabot closure.
2. No main-worktree dependency edit occurs before an exact measured scratch diff and explicit scope approval.
3. Do not merge stale Dependabot PR #16; recreate its `idna` change only as part of a current-master, verified lock resolution.
4. Keep ChromaDB separate from patchable dependency upgrades.
5. After every pushed dependency slice, verify the exact remote SHA, CI, Dependency Graph, and REST plus GraphQL alert state.
6. Stop and update this ledger when a resolver conflict, provider runtime incompatibility, or new advisory materially changes the campaign order.

## Current active step

**Phase 5 complete — ChromaDB accepted-but-open risk decision; monitor upstream patches and block non-local/server use pending a new decision.**
