# PromptManager — `prompt-validate` v1

**Status:** completed — 2026-09-21
**Owner:** PromptManager Team

## Product decision

Implement `prompt-validate <uuid|name> [--json]` as a provider-free, read-only technical check for one persisted prompt. It complements whole-catalog `catalog-check`: it is an inspect-before-change gate for a selected asset, not a lint engine, renderer invocation, model evaluation, or repair command.

## Confirmed seams and resulting scope

- `_resolve_prompt_reference()` already defines the canonical UUID-or-unique-exact-name lookup contract. Reuse it unchanged.
- `TemplateRenderer.extract_variables()` is the canonical local Jinja parser. Validate syntax with it and expose its detected variable names.
- A prompt currently stores no durable declared-variable schema and no durable model/type contract. Therefore v1 must **not** claim undeclared/unused-variable or model-compatibility checks.
- `related_prompts` is a persisted reference list; `PromptRepository.list()` supplies the local prompt inventory needed to verify it.
- Existing prompt fields are flexible dataclass values. V1 checks only locally meaningful structural conditions; importer deserialisation remains the write-time validation seam.

## CLI contract

```bash
python -m main --no-gui prompt-validate <uuid-or-exact-name>
python -m main --no-gui prompt-validate <uuid-or-exact-name> --json
```

JSON result shape:

```json
{
  "valid": true,
  "prompt": {"id": "...", "name": "..."},
  "variables": ["repository"],
  "summary": {"errors": 0, "warnings": 0},
  "issues": []
}
```

Text result prints the prompt identity, variables, summary, and deterministic issue lines.

Exit semantics:

- `0`: valid; warnings are reported but non-blocking.
- `4`: prompt not found.
- `5`: invalid/ambiguous prompt reference or one or more validation errors.
- `6`: repository loading/check failure.

## v1 checks

| Code | Severity | Condition |
| --- | --- | --- |
| `VAL001` | error | Prompt name is blank. |
| `VAL002` | error | Prompt description is blank. |
| `VAL003` | warning | Prompt body is empty or whitespace-only. |
| `VAL004` | error | Prompt body has invalid Jinja syntax. |
| `VAL005` | error | A `related_prompts` entry is not a UUID or points to no local prompt. |
| `VAL006` | warning | Tags contain blank values or duplicate values case-insensitively. |

## Explicit anti-scope

- No model/provider calls, rendering, persistence mutation, embedding checks, or `--fix`.
- No lint-style subjective prompt-writing advice; that belongs to future `prompt-lint`.
- No inferred declared-variable schema, unused-variable rule, or model compatibility assertion; current prompt storage has no authoritative data for those claims.
- No duplication of catalog-wide duplicate/chain checks from `catalog-check`.

## Execution order

1. Add failing CLI regressions for valid JSON evidence and error exit/output.
2. Add a compact provider-free report helper based on prompt and local prompt IDs.
3. Wire parser, command registry, deterministic text/JSON rendering, and documented exit semantics.
4. Verify focused tests, full entrypoint tests, Ruff, changed-path Pyright, help output, and diff check.
5. Update changelog and the CLI idea backlog with the delivered scope.

## Done criteria

- UUID/name lookup behavior matches `prompt-show` and `prompt-render`.
- Identical stored data yields identical report ordering.
- No provider, renderer execution, mutation, or embedding access occurs.
- Valid prompt variables are reported without inventing a schema contract.
- Warnings stay non-blocking; validation errors return `5`.

## Completion update

**Status: completed — 2026-09-21**

Delivered:

- Added `core/prompt_validation.py` with typed report/issue objects and deterministic `VAL001`–`VAL006` findings.
- Added `prompt-validate <uuid-or-exact-name> [--json]`, using the existing UUID-or-unique-exact-name resolver.
- Text output reports identity, discovered template variables, totals, and stable issue lines; JSON output exposes `valid`, `prompt`, `variables`, `summary`, and `issues`.
- No Jinja rendering, provider call, embedding access, repository mutation, or activity event is performed.
- `VAL003` and `VAL006` remain warnings; errors return `5`. Unknown/ambiguous prompt references preserve existing `4`/`5` behavior, and runtime inventory failure returns `6`.

Verified:

```bash
.venv/bin/pytest tests/test_main_entry.py::test_prompt_validate_command_reports_variables_and_warnings_as_json \
  tests/test_main_entry.py::test_prompt_validate_command_returns_error_for_invalid_template_and_reference -q
# 2 passed
```

Deferred: durable declared-variable schemas, unused-variable detection, model/type compatibility, subjective prompt linting, full catalog checks already covered by `catalog-check`, and automatic repair.
