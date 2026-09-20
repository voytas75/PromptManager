# PromptManager — Semantic `prompt-find` v1

Status: completed
Owner: PromptManager Team

## Goal

Make the existing `prompt-find` operator command execute natural-language semantic retrieval so:

```bash
uv run python -m main --no-gui prompt-find --limit 5 "score the instruction"
```

returns up to five best-fitting prompt assets, even when the query has no literal text match.

## Confirmed baseline

- `PromptManager.suggest_prompts(query, limit=...)` already provides semantic retrieval, intent hints, ranking, and bounded fallback through the configured embedding backend.
- `suggest` already exposes that domain seam, but `prompt-find` currently filters the SQLite prompt list by literal text over name, description, category, and tags.
- `prompt-find` currently has text and JSON render paths plus metadata filters; its current direct tests protect literal-match behavior only.

## Decision

Make implementation match the requested `prompt-find` contract by routing it through the existing `suggest_prompts` seam. Preserve the existing `--limit`, `--json`, category/tag/source/active filters, output shape, and no-result behavior. Filters apply after semantic ranking.

## Execution order

1. Add a RED CLI-entry regression where a natural-language query has no lexical overlap with the returned prompt but the suggestion seam supplies that ranked result.
2. Change only `run_prompt_find` to obtain candidates from `manager.suggest_prompts` and retain filters/rendering at the CLI boundary.
3. Update runtime help and developer command index to say that `prompt-find` is semantic/natural-language retrieval.
4. Run focused tests, direct no-provider CLI help smoke, lint/format, CI-scope Pyright, and diff checks.

## Out of scope

- Replacing or removing the existing `suggest` command.
- Changing embedding provider settings or making a live external provider call.
- New ranking models, persistence, API endpoints, commit, or push.

## Completion update

Completed 2026-09-20:

- Added a RED regression proving `prompt-find --limit 5 "score the instruction"` can return a ranked prompt with no literal query overlap.
- Routed `prompt-find` through the existing `PromptManager.suggest_prompts()` semantic retrieval seam.
- Preserved `--limit`, text/JSON rendering, and post-ranking category/tag/source/active filters.
- Updated runtime help and the developer CLI index to describe semantic natural-language retrieval.

Verification completed:

- focused `prompt-find` CLI regressions: 6 passed;
- focused `suggest_prompts` domain regressions: 6 passed;
- Ruff lint/format: passed;
- CI-scope Pyright (`main.py config models`): 0 errors;
- real isolated CLI smoke using deterministic embeddings: imported `Rubric Evaluator`, then `prompt-find --limit 5 "score the instruction"` returned it with exit 0 and empty stderr;
- `git diff --check`: passed.

Deferred unchanged: `suggest` remains available as the richer intent-diagnostic surface; this slice does not alter embedding/provider configuration or error-channel normalization.
