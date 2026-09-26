# Prompt-list CLI v1 — local execution ledger

Status: local implementation and project gates green; independent post-fix review found no blocker. Commit/push authorized; remote delivery and exact-SHA CI to be verified separately. No Windows acceptance claim.
Owner: PromptManager, subordinate to `docs/product-ssot.md` and the active near-term plan `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`.

## Decision and scope

A query-free catalog browse command is warranted because the current `prompt-find` requires a semantic query and `tag-show` requires a tag. Add `prompt-list` over the existing manager/repository; order locally by full modification timestamp (including microseconds) to compensate for SQLite `datetime()` precision loss; apply exact case-insensitive category/tag/source and optional active filters **before** a default-20, maximum-100 limit. Text shows identity/category/tags, JSON only bounded identity/metadata without body or embedding. `prompt-show` owns detail. No new index, ranking, storage, provider-backed evaluation, remote registry, or change to `prompt-find`.

Paths: `cli/parser.py`, `cli/commands.py`, `tests/test_main_entry.py`, `tests/test_prompt_list_cli.py`, `README.md`, `docs/README-DEV.md`, `docs/CHANGELOG.md`, this ledger and one compact `docs/STATUS.md` pointer. The existing manager can initialize local services; don't describe the whole entrypoint as immutable/no-bootstrap. No live provider or user catalog used for acceptance.

## Evidence

- Clean base: `master...origin/master` at `4fbb607776287150f9599b666e5620337c22d4af` before changes.
- RED: focused `pytest -k prompt_list_` returned 3 expected failures (`prompt-list` unknown to parser), 3 pre-existing parser-invalid checks passed.
- GREEN (focused): 125 tests in `tests/test_cli_help_contract_docs.py`, `tests/test_main_entry.py`, and `tests/test_prompt_list_cli.py`; 2 real-process isolated cases (installed and module entrypoints), no credentials/provider inputs. Real subprocess verifies selected record/order and JSON stdout/empty stderr.
- Independent review found SQLite `datetime(last_modified)` drops subsecond precision and could return an older prompt under `--limit 1`. Reproduced RED on both real entrypoints (2 failed); handler now sorts full timestamps before filters/limit, with a stable ID tie-break; both passed. Added an empty-catalog/boundary-limit check.
- GREEN (full, after fix): `pytest -n auto --cov=core --cov-report=term --cov-fail-under=80`: 1070 passed, 1 skipped, core coverage 81.67%; Ruff check and format across repo; project `pyright`: 0 errors; `uv lock --check`, earlier `uv build --wheel`, `git diff --check`; help for module and installed entrypoints. Direct file-scoped Pyright on the legacy unannotated `cli/commands.py` surfaces 123 inherited unknown-type errors and is not the project's configured gate. No paid model call.
- Security scan of added lines: no hardcoded secret, shell injection, eval/exec, pickle or formatted SQL pattern. Independent review blocker corrected; post-fix independent review found no blocker (real SQLite probe for subsecond, naive/aware timestamps and equal-time tie; 9 focused cases passed in its environment). Repo tests directly cover subsecond ordering on both entrypoints; naive/aware and equal-time cases have probe-only evidence.
- Local diff (tracked): README +15, commands +66, parser +27, changelog +1, developer index +1, status +4, entrypoint tests +144; untracked test 101 lines and ledger (this file). Do not infer CI, Windows, commit or push status from this local evidence.

## Follow-up boundary

Raw-text capture is already supported by `draft add TEXT|--body|--file|--from-stdin`, with an optional title and derived description; `prompt-add` accepts inline `--name --description --prompt-text` or JSON/file/stdin **JSON** input. Verified CLI help and the existing `tests/test_draft_cli.py` (16 passed) without writing to a user catalog. Therefore no new raw-text alias is justified solely by the competitor comparison. Change this decision only if observed operator use shows an actual missed capture or excess-step failure; preserve the explicit description requirement for promoted assets. Broader eval or remote sync require separate approval and measurable need.
