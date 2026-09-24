# PromptManager — controlled `prompt-edit` CLI v1

Status: implemented and published on `origin/master`; exact-SHA Quality Gates confirmed; no user catalog mutated
Product authority: `docs/product-ssot.md` (prompt metadata and controlled automation)
Related diagnostic contract: `docs/plans/2026-09-24-doctor-cli-contract-and-migration-v1.md`

## Confirmed state and decision

- `prompt-show <uuid> --json` exposes the `related_prompts` list; `doctor catalog` reports `CAT004` but is read-only. `prompt-tag` adds/removes tags, not arbitrary attributes. `catalog-import` matches a complete incoming record by *name*, so it is too broad to repair one field without further proof that other data remains unchanged. Before this slice, no `prompt-edit` CLI command existed.
- The invalid list `["[]"]` contains a literal string, not an empty list. Replacing it with `[]` is a data mutation, not a change in `doctor` interpretation.
- In this checkout, the reported prompt UUID already has `related_prompts: []` and no `CAT004`; its 93-prompt database is not the earlier 369-prompt database. Locate the correct selected catalog before any real repair.

**Decision:** expose one attribute-driven CLI command, not a separate command per field. **Do not** allow arbitrary `setattr`, SQL column names, or a generic whole-record rewrite. An explicit allowlist maps public attributes to their type/validation and write semantics; v1 enables only `related_prompts`, and other attributes are added to this command in separately verified slices. `doctor catalog` remains immutable.

## Public contract

```text
prompt-manager prompt-edit <prompt-uuid> set \
  --attr related_prompts --value '[]' \
  [--expect-value '["[]"]' --apply --backup-to PATH] [--json]
```

- `--attr` is a public field name, checked against a fixed allowlist. `--value` is exactly one JSON value; for `related_prompts` it must be a JSON array, not a quoted JSON string. `--json` controls the *result format*, not the value parser.
- In v1 only `related_prompts` is supported. `[]` means clear links; nonempty arrays must contain unique canonical UUID strings pointing to existing prompts in the **same** selected catalog. Reject `"[]"`, `["[]"]` as a new value, duplicate UUIDs, malformed UUIDs and missing targets. Preserve historical invalid list entries in the *expected* value so they can be matched and repaired, not silently normalized.
- UUID-only target in v1; names are ambiguous and are not a safe mutation key. Unknown UUID, missing catalog/config, unsupported field, invalid JSON/type/reference and persistence failure all fail closed with bounded error code/message and exit 2, without echoing input. Parser usage errors retain argparse exit 2.
- Default is a read-only **preview**. It shows selected attribute, prompt ID, before/after values and whether the field would change; `--json` emits one small result-only document (`command`, `prompt_id`, `attr`, `before`, `after`, `changed`, `applied: false`). No activity/version event, provider call, database/Chroma creation, or other side effect.
- Before an apply with changed data, the CLI rejects a catalog with a nonempty `-wal`/`-journal` sidecar, requires `--expect-value` and an explicit `--backup-to PATH` that does not already exist, and reserves the backup name exclusively. It uses SQLite's backup API and `PRAGMA integrity_check` before a single-column update. The comparison and write run under `BEGIN IMMEDIATE` plus an exact stored-value SQL precondition. The selected catalog is never created implicitly. A no-op with matching expected value does not create a backup. Keep GUI/other writers closed for this v1 maintenance operation; sidecar detection is a guard, not a general cross-process lock.
- The `related_prompts` repair intentionally does not change the prompt body, vector index, activity records, versions, `last_modified`, or other columns. The existing lifecycle versions a changed body (or missing history) and otherwise uses a full-row update; for metadata-only maintenance this direct single-column transaction avoids that broader rewrite. If future fields require lifecycle events/version snapshots, implement them in a separate slice instead of extending this direct SQL allowlist silently.
- Text mode prints `Preview`, `Applied`, or `No change`, attribute/UUID and before/after JSON arrays. `--json` prints exactly one result object to stdout and bounded code/message on stderr for a rejected edit (exit 2); neither error format echoes user input or prompt body. Unknown parser usage remains argparse exit 2.

**Strongest alternative:** edit/export/import the whole prompt through existing `catalog-import`. Reconsider only if an isolated test proves exact identity, unrelated-field, version/activity, provider and database-side-effect parity with a single-field edit; current name-based merge does not prove that.

## Bounded implementation slices

1. **Contract and write-seam proof:** inspect `PromptManager.update_prompt(..., refresh_derived_state=False)`, `core/repository/prompts.py`, activity/version semantics, SQLite WAL state, `main.py` startup, and installed CLI. Decide transactional CAS versus lifecycle and choose backup/exit contracts. Stop if no safe field-only path exists without broad manager/index startup.
2. **RED tests first:** real-process parser/help/preview/apply tests with temporary HOME/config/SQLite, starting from `["[]"]` and asserting preview leaves bytes untouched, `--apply --expect-value '["[]"]'` writes `[]`, read-back `prompt-show` shows `[]`, and `doctor catalog` no longer reports `CAT004` for the fixed prompt. Include valid UUID reference, clean/no-op, wrong expected value (no write), invalid new value, unsupported `--attr`, missing target/DB, absence of unrelated-column changes/provider/index calls, and JSON-only stdout with clear exit/stderr behavior.
3. **Minimal GREEN:** add `prompt-edit` parser/handler, a field allowlist+validator for `related_prompts`, and only the proven writer seam. Reuse diagnostic rules rather than redefining `CAT004`. Do not retrofit `prompt-tag` or `catalog-import`, and do not run apply against either user catalog as a test.
4. **Verify and document:** run focused subprocess tests, Ruff lint/format, configured strict Pyright, full provider-free pytest+coverage and `uv lock --check`; perform a disposable installed-CLI preview → apply → read-back → doctor smoke. Document help, examples, side effects, backup rule and errors in `docs/README-DEV.md` and `docs/CHANGELOG.md`; update this ledger with actual evidence. Commit/push require separate approval.

## Implemented slices and verification

1. Verified normal `update_prompt` may write the whole row (and version a changed body), whereas CLI metadata maintenance needs one column. A scratch SQLite probe showed backing up from the same connection under `BEGIN IMMEDIATE` blocks; using a second read connection while holding the write lock succeeded. No provider/index bootstrap is needed for this CLI path.
2. Added real-process tests before implementation (RED: `prompt-edit` initially rejected by parser). Tests now cover console script + `python -m main` help, JSON/text preview, apply/backup/readback/doctor, valid target UUID, stale expectation, invalid list/unsupported field, missing catalog/ID, backup collision, no-op and pending WAL. A test checks no version/activity additions and unchanged description/body.
3. Implemented `cli/prompt_edit.py`, dedicated parser/dispatch in installed console script and module entry point, without routing through the manager. First and only allowlisted field is `related_prompts`. The existing `doctor` code remains unchanged.
4. Final provider-free full suite after guard and test refinements: `998 passed, 1 skipped`, core coverage `81.66%` (`QT_QPA_PLATFORM=offscreen .venv/bin/pytest -n auto --cov=core --cov-report=term-missing --cov-fail-under=80`). Ruff lint+format, full strict Pyright, `uv lock --check` and `git diff --check` passed. Real-process tests use the installed console script and disposable catalog; no real user catalog was edited.
5. Delivered feature commit `2cc3e4c14f1e526196ef05cdc85026cea230f8c2` to `origin/master`; local `HEAD`, tracking branch and remote SHA matched. Exact-SHA GitHub Quality Gates [run 36063036058](https://github.com/voytas75/PromptManager/actions/runs/36063036058) succeeded (Ruff, Pyright, Pytest and clean tree). This ledger-only closeout commit and its CI: **to verify**.

## Done and stop conditions

Done means an installed real CLI can repair only the selected field of a synthetic prompt with explicit precondition, rejects stale/invalid operations with zero writes, returns readable text or one machine-readable JSON result, keeps other columns and diagnostic semantics unchanged, and has a verified rollback/backup posture and repo-facing documentation. This plan is not authorization to change production/user data.

**Next slice after local verification:** identify the correct 369-prompt catalog in its own environment, check `prompt-show --json` and preview there, then obtain explicit approval before applying to that user catalog. Do not implement unrestricted attribute editing or automatic `doctor --fix`.
