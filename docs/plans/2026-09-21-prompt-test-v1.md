# PromptManager — `prompt-test` v1

**Status:** in progress
**Owner:** PromptManager Team

## Product decision

Implement `prompt-test <uuid|name> --suite PATH [--json]` as a deterministic, provider-free regression runner for a persisted prompt template. It verifies local Jinja rendering against explicit input/output fixtures; it does not claim to evaluate model behaviour.

## Confirmed baseline

- PromptManager has `prompt-render`, which locally renders a prompt with JSON variables, but no persisted test-suite/test-case model or test-run history.
- `benchmark` executes configured models and is therefore a distinct, provider-backed operational surface.
- A durable suite repository, named suite commands, test histories, model-response assertions, and evaluator scores do not exist yet.

## v1 CLI contract

```bash
python -m main --no-gui prompt-test <uuid-or-exact-name> --suite tests.json
python -m main --no-gui prompt-test <uuid-or-exact-name> --suite tests.json --json
```

Suite JSON shape:

```json
{
  "cases": [
    {
      "id": "T01-summary",
      "variables": {"repository": "PromptManager"},
      "expected": "Review PromptManager."
    }
  ]
}
```

Each case renders the selected prompt using `variables`, then compares the rendered text to `expected` exactly. Case IDs must be unique and non-empty. `cases` must be non-empty.

JSON result shape:

```json
{
  "ok": true,
  "prompt": {"id": "...", "name": "..."},
  "suite": {"path": "tests.json", "case_count": 1},
  "summary": {"total": 1, "passed": 1, "failed": 0},
  "cases": [{"id": "T01-summary", "status": "passed", "error": null}]
}
```

Exit semantics:

- `0`: all cases pass.
- `4`: prompt not found.
- `5`: ambiguous prompt reference, invalid suite, or one or more failed cases.
- `6`: prompt/suite load failure outside normal suite validation.

## Boundaries

- No provider/model calls, persistence mutation, test-result history, embedding calls, or activity events.
- No fuzzy/LLM judging, latency/cost checks, or benchmark integration.
- No persisted/named suite catalog yet: `--suite PATH` is explicit to make suite provenance visible and reproducible.
- Exact output assertion only in v1; richer matchers or JSON-path assertions need an explicit future contract.

## Execution order

1. Add RED entrypoint regressions for a mixed pass/fail suite and malformed suite input.
2. Add typed provider-free suite parser/runner over `TemplateRenderer`.
3. Wire parser, command registry, text/JSON reports, and exit codes.
4. Verify focused tests, full suite, Ruff, narrow Pyright, help, and docs coverage.

## Done criteria

- Identical prompt + suite input gives deterministic case ordering and results.
- The command resolves prompts through the existing UUID-or-unique-exact-name contract.
- Failures identify individual case IDs without printing expected or rendered prompt bodies by default.
- No provider or persistence operation occurs.

## Completion update

**Status: completed — 2026-09-21**

Delivered:

- Added `core/prompt_testing.py` with typed suite cases, bounded case results, strict JSON parser, and local exact-output runner.
- Added `prompt-test <uuid-or-exact-name> --suite PATH [--json]` using the established prompt resolver.
- Suite schema is explicit and reproducible: a non-empty `cases` list; each case has unique non-empty `id`, optional object `variables`, and string `expected`.
- Report output includes suite path, total/pass/fail counts, and case IDs/statuses; it deliberately omits rendered and expected text.
- The runner calls only `TemplateRenderer.render`; it makes no provider, embedding, persistence, history, or activity call.
- Invalid suites and test failures return `5`; all-pass suites return `0`.

Verified:

```bash
.venv/bin/pytest tests/test_main_entry.py::test_prompt_test_command_reports_mixed_fixture_results_as_json \
  tests/test_main_entry.py::test_prompt_test_command_reports_invalid_suite_as_json -q
# 2 passed

.venv/bin/ruff check core/prompt_testing.py cli/parser.py cli/commands.py tests/test_main_entry.py
.venv/bin/ruff format --check core/prompt_testing.py cli/parser.py cli/commands.py tests/test_main_entry.py
.venv/bin/pyright core/prompt_testing.py cli/parser.py tests/test_main_entry.py
# passed / 0 errors
```

Deferred: stored/named suite management, suite discovery/listing, durable run history, non-exact matchers, provider response tests, evaluator scoring, cost/latency checks, and integration with `benchmark`/future `prompt-evaluate`.
