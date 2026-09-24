# PromptManager — documentation claim closeout v1

Status: completed locally; delivery authorized, exact-SHA CI pending
Date: 2026-09-24
Baseline: `603856c34312103e919af2e48e15e3dc3c2daa32` on clean `master...origin/master`
Product authority: `docs/product-ssot.md`
Near-term priority authority: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Decision and scope

Close the three confirmed documentation findings from the read-only audit without changing behavior, product priorities, package version, CI, or historical evidence. Edit only `docs/README-DEV.md`, `docs/session-restart-brief-2026-04-06-slice-guidelines.md`, `docs/CHANGELOG.md`, and this ledger. No provider calls or user-data changes. Commit and push were separately authorized after local closeout.

## Ordered corrections and acceptance

1. **Execution boundary** — Replace `offline-only runs` in the developer guide with precise `without web search` wording; explicitly retain model/provider execution requirement. Check against `cli/commands.py` and `core/prompt_manager/chains.py`, then run focused contract assertions and `git diff --check`.
2. **Authority / restart navigation** — Keep the April brief and its dated evidence, but demote its active/current/superior restart language; point first to the product SSOT, current near-term plan and status sink. Check no unconditional instructions still treat the brief as the current authority, local link targets resolve, and `git diff --check` passes.
3. **Release history** — Consolidate both `0.23.0` sections under one dated heading without dropping entries or changing other releases. Programmatically compare old and new version-0.23.0 bodies and heading uniqueness; run `git diff --check`.
4. **Final review** — Independent read-only review of the changed claims, local Markdown/HTML target scan, authority-phrase scan, exact changed/untracked file inventory and diff. Confirm only these four Markdown files changed. Prove remote delivery and exact-SHA CI separately after the authorized push.

## Progress ledger

- 2026-09-24 — Plan created from clean `603856c` baseline. Items 1–4 pending.
- 2026-09-24 — Item 1 completed locally: `docs/README-DEV.md` now limits `--no-web-search` to web enrichment and states that chain execution still needs a model/provider. Static contract assertions against the guide, `cli/commands.py`, and `core/prompt_manager/chains.py` passed; `git diff --check` passed. Next: item 2 authority/navigation.
- 2026-09-24 — Item 2 completed locally: April restart brief is explicitly historical, its dated evidence remains, and its decision path points to the active product SSOT, near-term plan and status. Authority assertions, target existence checks and `git diff --check` passed. Next: item 3 release headings.
- 2026-09-24 — Item 3 completed locally: removed the duplicate `0.23.0` release heading while retaining the complete nonblank release text and all other headings. An initial overly strict byte-for-byte check failed on blank-line removal; the corrected nonblank-content comparison passed, and all 80 numbered release headings plus `Unreleased` are unique. `git diff --check` passed. Next: independent final review and link/authority/scope checks.
- 2026-09-24 — Independent review found two in-scope corrections before closeout: the guide conflated workspace-only web-context condensation with the Chain path, and this ledger called `Unreleased` a release heading. Both are corrected against `gui/controllers/execution_controller.py` and `core/prompt_manager/chains.py`; final verification below. Historical `0.22.3` changelog wording remains a dated record, not active guidance.
- 2026-09-24 — Item 4 completed locally: focused provider-free `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_cli_help_contract_docs.py tests/test_prompt_chain_cli.py tests/test_offline_gui_startup.py` passed (31 tests). Final claim assertions confirmed the CLI/chain execution boundary, workspace-only web-context condensation, current authority pointers, unchanged nonblank `0.23.0` release content, and 80 unique numbered release headings plus `Unreleased`. All 22 Markdown/HTML links or images in the four changed files resolve; `git diff --check` passed. Exact inventory: three modified Markdown files (`docs/README-DEV.md`, `docs/session-restart-brief-2026-04-06-slice-guidelines.md`, `docs/CHANGELOG.md`) and this one untracked ledger. No application code or tests changed.
- 2026-09-24 — User separately authorized commit and push. Delivery and exact-SHA CI are pending verification; no remote success is implied by this note.

## Completion / next decision

All four items are complete locally. The independent review's in-scope findings were reconciled and the final claims and changed-file inventory rechecked. Historical `docs/CHANGELOG.md` text under `0.22.3` still records its then-current “offline-only” wording; it is not current operator guidance. Commit/push have been authorized; delivery and exact-SHA CI remain to verify.

Separate follow-up candidates, **not authorized by this slice**:
- First-run guides: distinguish provider-free `doctor` from headless `--print-settings`/GUI startup requiring an existing JSON config (isolated `.env.example`-only non-interactive `--print-settings` exited 2; with copied JSON config exited 0). Also disclose the template's JSON-over-env model/embedding selections in the main README.
- Legacy `catalog-check`: qualify the read-only *audit* versus manager initialization that can create local SQLite/Chroma files; `doctor catalog` is the immutable inspection route.
- Import sample: `scenario` is not mapped by `core/catalog_importer.py`; decide on a supported field or remove the misleading example. Align `pyproject.toml`'s Python 3.12 classifier with `requires-python = ">=3.13"` in a separate metadata slice.
- Historical status hygiene: recheck the delivery-pending labels in the Linux GUI, local-first GUI, and Recent reopen ledgers and the “Latest” headings in `docs/STATUS.md` against Git history before changing them. Keep the Dependabot campaign separate.
