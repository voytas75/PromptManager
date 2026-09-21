# PromptManager — `prompt-template-list` v1

**Status:** completed — 2026-09-21
**Owner:** PromptManager Team

## Product decision

Implement `prompt-template-list [--json]` as a read-only, manager-free inspection command for all built-in LiteLLM workflow templates. The text format must make every full template readable in a terminal with explicit Unicode borders, identity, provenance, character/line counts, and numbered body lines.

## Confirmed seams

- `prompt_templates.py` is the canonical source for the fixed template key order, labels, descriptions, and default text.
- `PromptManagerSettings.prompt_templates` holds optional user overrides; `core.factory.build_prompt_manager()` treats non-empty configured text as the runtime override source.
- Template reading needs settings but does not need the repository, embeddings, prompt manager, provider credentials, or a model call.
- `prompt_templates.py` is packaged with the wheel, so the command remains available from the installed console entrypoint.

## CLI contract

```bash
python -m main --no-gui prompt-template-list
python -m main --no-gui prompt-template-list --json
```

Text output:

- starts with the effective template total;
- renders every template in `PROMPT_TEMPLATE_KEYS` order;
- gives each record a `╔═ … ═╗` heading and `╚═…═╝` boundary;
- includes key, label, default/override source, description, character count, line count, and numbered/wrapped body lines.

JSON output:

```json
{
  "templates": [
    {
      "key": "name_generation",
      "label": "Prompt name suggestions",
      "description": "...",
      "source": "default",
      "text": "...",
      "characters": 123,
      "lines": 1
    }
  ]
}
```

## Scope and boundaries

- Include all six canonical workflow templates and apply a non-empty configured override only for its known key.
- `source` is `default` when no materially different override is active and `override` otherwise.
- `--json` exposes full text deliberately for automation; text mode exposes full text deliberately for human review.
- No persistence mutation, provider/model call, workflow execution, embedding access, manager construction, activity event, or template editing.
- No `--set`, `--reset`, filtering, secrets inspection, or runtime health claims.

## Execution order

1. Add a typed effective-template report helper and focused tests for source selection/order.
2. Add parser/manager-free handler, readable renderer, JSON output, and entrypoint tests.
3. Update CLI index, changelog, and plan.
4. Verify focused tests, help, full suite, release gate parity, wheel inclusion, and diff hygiene.

## Done criteria

- One command shows every canonical template exactly once and in canonical order.
- Default/override provenance is observable without exposing settings other than template text already selected for output.
- Terminal output remains visibly structured for multi-line templates.
- The command can run when repository/provider initialization would be unavailable.

## Completion update

**Status: completed — 2026-09-21**

Delivered:

- Added `core/prompt_template_listing.py`, deriving records exclusively from canonical `prompt_templates.py` data and validated setting overrides.
- Added manager-free `prompt-template-list [--json]`; it shows six canonical effective templates in source order.
- Text output uses Unicode-bordered records, explicit source/metrics/description metadata, and numbered wrapped body lines; JSON exposes full ordered records.
- The entrypoint loads settings but does not construct `PromptManager`, repository, embeddings, or provider clients for this command.

Verified:

```bash
.venv/bin/pytest -q
# 867 passed, 1 skipped
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pyright main.py config models
# all passed
```

Deferred: template editing/reset CLI operations, template filtering, exposing arbitrary settings, provider health assertions, and runtime workflow execution.
