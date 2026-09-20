# PromptManager — Operational Documentation Currentness v1

Status: completed
Owner: PromptManager Team

## Goal

Reconcile active operational documentation with the delivered CLI contract changes at `d443e8b513ca7485cbddb47d3a0a4aa7c58b51b3`, without rewriting historical evidence.

## Confirmed baseline

- Current local and remote `master` both resolve to `d443e8b513ca7485cbddb47d3a0a4aa7c58b51b3`.
- Exact-SHA Quality Gates completed successfully for that revision.
- The locally verified current full gate, with stale `PROMPT_MANAGER_CONFIG_JSON` unset, is `839 passed, 1 skipped`, coverage `80.39%`, plus Ruff/format and CI-scope Pyright green.
- The active `docs/STATUS.md` still names a 2026-09-03 commit/checkpoint as current.
- The 2026-09-03 CLI retrieval decision calls `prompt-find` literal/offline-only, while the delivered 2026-09-20 contract intentionally makes it semantic.
- `docs/CHANGELOG.md` omits the three delivered CLI contract slices, and `docs/README-DEV.md` calls the broader `nox -s all` run CI-equivalent even though it invokes full Pyright while CI uses `pyright main.py config models`.

## Execution order

1. Add one current checkpoint to `docs/STATUS.md`; retain prior results as historical.
2. Mark the superseded retrieval decision as historical and link its successor.
3. Record the delivered semantic retrieval and bounded prompt-show JSON contracts in `docs/CHANGELOG.md`.
4. Correct the Nox/CI wording in `docs/README-DEV.md`.
5. Verify local links, authority wording, diff hygiene, and current documented quality commands.

## Out of scope

- Runtime, test, workflow, dependency, security, or product-roadmap changes.
- Revising historical technical evidence beyond authority/currentness labels.
- Commit or push.

## Completion update

Completed 2026-09-20:

- Added the current `d443e8b` revision, delivered CLI contracts, exact-SHA CI result, and current local quality evidence to `docs/STATUS.md`.
- Kept prior 2026-09-02/03 results as explicitly historical checkpoints.
- Marked the superseded literal-only retrieval decision as historical and linked its semantic successor.
- Added the delivered semantic `prompt-find` and bounded `prompt-show --json --full` contracts to `docs/CHANGELOG.md`.
- Corrected `docs/README-DEV.md` so `nox -s all` is documented as a broader local quality run, not exact CI parity.

Verification completed:

- local links in current authority/docs files: all resolved;
- authority/currentness phrase scan: passed;
- `git diff --check`: passed;
- current full local gate with stale explicit config override unset: `839 passed, 1 skipped`, coverage `80.39%`, Ruff/format and CI-scope Pyright passed;
- exact-SHA Quality Gates for `d443e8b` succeeded.

Deferred unchanged: runtime, workflow, dependency, security, product-roadmap, commit, and push changes.
