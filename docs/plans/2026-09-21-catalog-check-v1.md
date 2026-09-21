# PromptManager — `catalog-check` v1

**Status:** in progress
**Owner:** PromptManager Team

## Product decision

Implement `catalog-check` as a read-only, deterministic integrity pass for prompt assets and stored prompt chains. It is the first practical step from artifact management toward a software-like quality loop, but it must remain asset-first and provider-free.

## Confirmed implementation seams

- `PromptRepository.list()` is the canonical prompt inventory.
- `PromptManager.list_prompt_chains(include_inactive=True)` exposes stored chains and their prompt references.
- `TemplateRenderer.extract_variables()` is the canonical Jinja syntax parser.
- Embedding diagnostics already distinguish missing vectors from storage consistency; v1 avoids live backend/Chroma diagnostics to keep the command deterministic and provider-free.

## v1 checks

| Code | Severity | Condition |
| --- | --- | --- |
| `CAT001` | warning | More than one prompt uses the same exact non-empty name. |
| `CAT002` | warning | More than one prompt has the same non-empty normalized body. |
| `CAT003` | error | Prompt body has invalid Jinja template syntax. |
| `CAT004` | error | A `related_prompts` entry is not a UUID or points to a missing prompt. |
| `CAT005` | warning | Prompt has no stored embedding vector. |
| `CAT006` | error | A stored chain step references a missing prompt. |

## CLI contract

```bash
python -m main --no-gui catalog-check
python -m main --no-gui catalog-check --json
```

Text output: checked counts, error/warning totals, then deterministic issue lines.
JSON output: checked counts plus structured issue records.

Exit semantics:
- `0`: no errors (warnings are reported but non-blocking)
- `5`: one or more integrity errors
- `6`: inventory/check loading failed

## Explicit anti-scope

- No mutation, cleanup, repair, provider calls, embedding generation, or Chroma inspection.
- No fuzzy similarity/semantic duplicate detection.
- No lint-style subjective writing advice.
- No `--fix`, `--dry-run`, or model compatibility analysis in v1.

## Execution order

1. Add provider-free CLI regressions for clean, warning/error, and JSON paths; observe RED.
2. Add a compact core report/checker using existing prompt, chain, and template seams.
3. Wire parser, command registry, text/JSON rendering, and exit codes.
4. Run focused tests, Ruff, changed-path Pyright, root help check, and diff check.
5. Update this ledger and changelog after verified delivery.

## Done criteria

- Check results are deterministic for identical repository data.
- No external provider, embedding generation, or repository mutation occurs.
- Prompt duplicate/template/lineage and chain-reference integrity cases are covered.
- Warnings do not fail the command; errors return exit code `5`.

## Completion update

**Status: completed — 2026-09-21**

Delivered:

- Added provider-free `core/catalog_check.py` and `catalog-check [--json]`.
- Added the six v1 issue codes (`CAT001`–`CAT006`) described above.
- Kept the command read-only: it loads prompts/chains, uses Jinja parsing only, and does not call providers, Chroma, embedding generation, or mutation paths.
- Text mode gives summary counts plus deterministic issue lines; `--json` provides structured `summary` and `issues` records.
- Error-level findings return `5`; warnings stay non-blocking with exit `0`.

Verified:

```bash
.venv/bin/pytest tests/test_main_entry.py::test_catalog_check_reports_integrity_issues_and_json_payload \
  tests/test_main_entry.py::test_catalog_check_reports_a_clean_catalog -q
# 2 passed

.venv/bin/ruff check core/catalog_check.py cli/parser.py cli/commands.py tests/test_main_entry.py
.venv/bin/ruff format --check core/catalog_check.py cli/parser.py cli/commands.py tests/test_main_entry.py
.venv/bin/pyright core/catalog_check.py cli/parser.py tests/test_main_entry.py
# passed / 0 errors

.venv/bin/python -m main --help
.venv/bin/python -m main catalog-check --help
git diff --check
# passed
```

Deferred: stale/orphaned Chroma inspection, invalid metadata beyond model deserialization, duplicate-content fuzziness, whole-catalog template-schema contracts, cleanup/repair, and any provider-backed checks.
