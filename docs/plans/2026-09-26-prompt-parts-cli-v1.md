# PromptManager — Prompt Parts CLI v1

Status: implemented and verified locally; no commit/push. Product authority: `docs/product-ssot.md` (assets first, automation subordinate). Starting revision: `f306aa2`. Predecessor: `docs/plans/2026-09-26-prompt-parts-canonical-snippet-slice.md`.

## Goal and verified baseline

Give an operator or local agent a provider-free CLI to find, inspect, create, change and remove the **same** Prompt Parts that the GUI stores in SQLite `response_styles`. `ResponseStyle.snippet` is the canonical fragment body, independent of description, formatting instructions and examples. The GUI's `version` is a mutable label, **not** immutable revision history. The repository exposes CRUD but its constructor bootstraps/migrates schema; the CLI therefore uses a separate existing-catalog seam. Both `prompt-manager` and `python -m main` now dispatch `prompt-part` before the full application runtime. The repository search matches name, description and classification, not snippet; the CLI's literal search also covers snippet.

## Delivered command contract

Use one `prompt-part` family (singular, matching `note`/`draft`), human text by default and result-only `--json` on every leaf. Name/classification are labels, **never identifiers**: mutations and `show` require a full canonical UUID; duplicate names are allowed. All operations use the selected configured SQLite catalog, not a second store.

- `prompt-part [--limit N] [--all] [--json]`: active parts by default, sorted `name COLLATE NOCASE, id`, limit default 20 and range 1–100. `--all` includes inactive. Return bounded list rows: ID, name, classification, active flag, short snippet preview; no full text by default.
- `prompt-part find <literal> [--limit N] [--all] [--json]`: case-insensitive per SQLite ASCII `LIKE`, literal `%`, `_` and escape character, matching name, classification, description **and snippet**. Same ordering and compact results. No semantic retrieval or ranking.
- `prompt-part show <uuid> [--json]`: full canonical snippet and metadata (description, type, tone, voice, formatting, guidelines, tags, examples, active flag, editable version label, UTC timestamps). Text output separates raw snippet clearly; JSON returns a bounded explicit record, not undocumented `ext*` storage columns. Treat text as user data, not as a template to execute.
- `prompt-part add --name NAME [--part LABEL] [--description TEXT] [--body TEXT | --file PATH | --from-stdin] [--json]`: require exactly one nonblank UTF-8 snippet source; text limit 1 MiB, file must exist and fit. `--part` defaults to `Response Style`; name must be nonblank; description may be blank. Create a UUID and timestamps in existing table. Return ID and saved record/receipt; no provider, embedding or prompt mutation.
- `prompt-part edit <uuid> --expect-modified ISO8601 [--name NAME] [--part LABEL] [--description TEXT] [--active | --inactive] [--body TEXT | --file PATH | --from-stdin] [--json]`: require at least one explicit change; snippet source optional, but if present exactly one and nonblank. Preserve all unspecified fields, particularly tone, voice, format instructions, guidelines, examples, tags, metadata, `version`, `created_at`, and `ext*`; do not rebuild a partial `ResponseStyle` from CLI defaults. Allow `--description ''` to clear description. Keep GUI's mutable version label unchanged in v1. Reject stale `last_modified` under the same SQLite write transaction, not by a separate preflight. A matching no-op need not update timestamp.
- `prompt-part delete <uuid> --expect-modified ISO8601 [--yes] [--json]`: hard-delete the existing part. TTY confirmation defaults No; `--yes` required for non-TTY and JSON. Check the expected timestamp and target under the write transaction; a stale/missing target leaves data unchanged. No extra deletion from prompts/runs because no reference-composition contract exists.

CLI v1 deliberately leaves extended metadata **readable and preserved**, but not editable via new CLI flags; adding full metadata editing or structured import/export is a later slice. `--json` is an output selector, not JSON input. No shell-unsafe raw snippet in errors.

Implemented local contract: `add` and changed `edit` return `part`, while `delete` returns `id`; list/find return `parts`. Deletion requires `--yes` in JSON or non-TTY mode. The CLI does not create a missing catalog, run schema migrations, load providers/Chroma, or promise a snapshot across independent GUI/CLI processes.

### Machine and failure contract

On success, stdout contains one JSON object with `ok: true` and one command-specific key (`parts`, `part`, or `id`); stderr is empty. List/find items are compact; `show` and an explicit mutation receipt may include complete snippet text. On validation/parse/runtime failure, stdout is empty, stderr contains one sanitized JSON error (`ok: false`, stable `error.code` and generic `message`) in JSON mode, exit nonzero. Root/family/leaf `--help` exits 0 on stdout. Text-mode failures also avoid echoing body, raw SQL exception, catalog path, credential or supplied invalid token. Exact fields and error channels are exercised in real-process tests; no broad external compatibility commitment beyond this v1 slice is implied.

Illustrative output shapes: `{"ok":true,"parts":[{"id":"<uuid>","name":"System policy","prompt_part":"System Instruction","is_active":true,"preview":"Follow..."}]}`, `{"ok":true,"part":{"id":"<uuid>","name":"System policy","snippet":"Follow these instructions.","description":"Usage notes","prompt_part":"System Instruction","is_active":true,"version":"1.0","created_at":"<ISO8601>","last_modified":"<ISO8601>"}}`, and stderr `{"ok":false,"error":{"code":"PART_NOT_FOUND","message":"Prompt part UUID was not found."}}`. The `show` record also includes the remaining named supporting fields listed above; its exact JSON shape is verified in process tests.

No default read/write may create a missing DB, run migrations, initialize Chroma, import LiteLLM/GUI, call a provider, or load untrusted template code. Open the selected existing DB in SQLite `mode=ro` for reads, `mode=rw` for writes; validate the `response_styles` columns before a query. An older catalog lacking `snippet` fails closed with an actionable `CATALOG_MIGRATION_REQUIRED` error; migrate only through the already-supported repository/GUI startup on a backed-up catalog, never implicitly during a CLI read. Writes use parameterized SQL and explicitly checked transaction outcomes. Document SQLite/WAL behavior rather than claiming a cross-process snapshot that was not proven.

## Ordered implementation slices (all complete locally)

1. **Read-only foundation — complete.** RED subprocess tests showed missing command; installed/module entrypoints, help, list/find/show, active filtering, duplicate names, literal wildcard matches, Unicode, JSON channels, missing DB and legacy schema now pass. `cli/prompt_part.py` opens existing SQLite with `mode=ro`, not through `PromptRepository.__init__`.
2. **Create — complete.** RED tests for one exact source, invalid/oversized/invalid UTF-8, missing catalog; add→show and GUI/repository read-back preserve UUID, snippet (including trailing newline) and independent description. No schema or embedding changes.
3. **Edit and delete — complete.** RED tests for omitted-field preservation, empty description, inactive flag, stale/invalid timestamp, malformed UUID, no-op, GUI update between CLI calls, locked writer, non-TTY delete refusal and confirmed delete. `BEGIN IMMEDIATE` and raw `last_modified` comparison happen under one writer lock. Only requested fields change; supporting/extension columns survive. Rejected mutations retain records.
4. **Contract and closeout — complete locally.** Installed/module help, text/JSON and sanitized errors, malformed stored records, import-only provider boundary, and full add→list/find/show→edit→delete on isolated catalogs were exercised. `docs/README-DEV.md` and `docs/CHANGELOG.md` match the delivered command set. Full verification evidence follows. No product SSOT update, commit, push or Windows-native acceptance in this scope.

## Local evidence and delivery boundary

- `tests/test_prompt_part_cli.py` uses disposable SQLite catalogs and subprocesses for both entrypoints; 23 focused tests passed after a RED/GREEN cycle, including lifecycle and locked-writer cases.
- Final post-change `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q` — **1061 passed, 1 skipped**. `.venv/bin/ruff check .`, changed-file `.venv/bin/ruff format --check`, `.venv/bin/pyright` (0 errors), and `git diff --check` passed in the same gate command. Import-only check of `cli.prompt_part` found none of `litellm`, `chromadb`, `PySide6`, `gui`, `core.prompt_manager` loaded. No provider-backed live calls were made.
- The actual user's GUI catalog was not mutated. External delivery (`git commit`/push/CI) and native Windows GUI acceptance remain to verify only if separately requested.

## Risks and stop conditions

- Legacy SQLite schema, malformed rows or concurrent GUI writes: fail closed, show no raw stored text in errors; test a disposable migrated copy, not the user's selected catalog.
- Existing `ResponseStyle.from_record()` provides permissive defaults; reject incompatible/malformed stored rows at the CLI boundary rather than silently inventing identities or timestamps.
- Existing repository updates write every column. Prefer a conditional field-targeted transaction for CLI edits; do not mistake a preliminary `get` plus unconditional `update` for concurrency safety.
- If read-only startup imports providers or emits config noise, repair that boundary before adding mutations. If edit/delete cannot preserve GUI data or honor concurrency without wider schema/lifecycle changes, stop after the read-only slice and request a scope decision.

## Decision check

Strong alternative: mirror GUI CRUD immediately via full `PromptManager` methods. That is less CLI SQL but boots/migrates the repository on reads and can trigger unrelated service initialization. Prefer the early provider-free existing-catalog seam; revisit if a measured lightweight manager/repository read-only facade can guarantee no bootstrap, migration, provider, or parallel business rules. The first concrete signal is a real isolated subprocess read with no DB/index creation and clean JSON channels.
