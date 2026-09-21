# PromptManager — `prompt-lint` v1

**Status:** completed — 2026-09-21
**Owner:** PromptManager Team

## Product decision

Implement `prompt-lint <uuid|name> [--json]` as a read-only, deterministic, provider-free advisory check for one persisted prompt. It helps an operator notice simple maintainability and clarity risks without claiming technical invalidity, model quality, or business correctness.

## Confirmed seams and scope

- `_resolve_prompt_reference()` remains the canonical UUID-or-unique-exact-name resolver.
- `TemplateRenderer.extract_variables()` provides the local Jinja variable inventory, but lint must tolerate parse failures rather than duplicate `prompt-validate` syntax diagnostics.
- Prompt storage has no authoritative declared-input schema, acceptance rubric, or semantic evaluator. Therefore v1 cannot claim a variable is unused, a prompt is vague in a linguistic sense, or that a generated answer would be correct.
- `prompt-validate` owns technical errors; `prompt-test` owns exact local rendering regressions; `prompt-evaluate` remains the future provider/dataset quality lane.

## CLI contract

```bash
python -m main --no-gui prompt-lint <uuid-or-exact-name>
python -m main --no-gui prompt-lint <uuid-or-exact-name> --json
```

JSON shape:

```json
{
  "clean": false,
  "prompt": {"id": "...", "name": "..."},
  "metrics": {"body_characters": 210, "body_lines": 9, "variables": ["audience"]},
  "summary": {"warnings": 2},
  "issues": [{"code": "LINT001", "severity": "warning", "message": "..."}]
}
```

Exit semantics:

- `0`: lint completed, including when warnings are reported.
- `4`: prompt is not found.
- `5`: prompt reference is ambiguous.
- `6`: local lint processing failure.

## Deterministic v1 checks

| Code | Condition | Advisory |
| --- | --- | --- |
| `LINT001` | Non-empty description has fewer than 20 characters. | Expand the description so a future operator can identify the prompt purpose. |
| `LINT002` | Non-empty body has no imperative/action cue from a fixed, documented vocabulary. | State the requested task or action explicitly. |
| `LINT003` | Body has 12+ non-empty lines but no Markdown heading or numbered-list marker. | Add structure to make a long prompt easier to review and modify. |
| `LINT004` | A non-empty, normalized body line is repeated. | Consolidate repeated instruction lines to reduce maintenance ambiguity. |
| `LINT005` | The body uses Jinja variables but the description does not mention any input/variable context. | Document expected input context in the description. |

The report includes body character/line counts and detected variables as evidence. Invalid Jinja is represented as `template_parse_error` in metrics, not a lint warning: `prompt-validate` remains the owner of syntax validity.

## Explicit anti-scope

- No model/provider calls, rendering, persistence mutation, activity events, embeddings, history reads, or `--fix`.
- No subjective claim that an instruction is factually correct, sufficiently detailed, safe, or semantically non-vague.
- No technical validation duplication (blank fields, invalid syntax, references, tags) and no test/evaluation execution.
- No comparison to previous versions in v1; this needs a bounded version-baseline policy.

## Execution order

1. Add core lint report/helpers and focused unit tests for stable findings and parse-error boundary.
2. Add CLI parser, resolver reuse, text/JSON renderer, and tests for warnings and reference errors.
3. Document the command in the developer CLI index, changelog, and living idea backlog.
4. Run targeted tests, full suite, Ruff, format, narrow Pyright, CLI help, and diff checks.

## Done criteria

- The command makes no provider, renderer, repository-write, or activity-ledger call.
- Repeated input yields identically ordered findings.
- Warnings are advisory and never turn a successful lint into exit `5`.
- `prompt-validate`, `prompt-test`, and future `prompt-evaluate` boundaries stay explicit in docs and output.

## Completion update

**Status: completed — 2026-09-21**

Delivered:

- Added `core/prompt_linting.py` with typed, deterministic `LINT001`–`LINT005` advisory findings.
- Added `prompt-lint <uuid-or-exact-name> [--json]` using the established UUID-or-unique-exact-name resolver.
- JSON reports `clean`, identity, bounded metrics, warning totals, and stable issue records; text reports the same evidence for operators.
- Warnings return exit `0`; resolver failures retain exit `4`/`5`, and local lint failure returns `6`.
- No provider call, rendering, persistence mutation, execution/history access, embedding access, or activity event is made.

Verified:

```bash
.venv/bin/pytest -q
# 864 passed, 1 skipped
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pyright main.py config models
# all passed
```

Deferred: linguistic/semantic vagueness, conflicting constraints, unused-variable assertions, version-length deltas, model-backed judgement, automatic repair, and all provider/dataset evaluation.
