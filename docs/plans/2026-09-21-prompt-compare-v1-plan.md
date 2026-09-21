# PromptManager — `prompt-compare` v1 plan

**Status:** proposed / not implemented
**Owner:** PromptManager Team

## Decision

Design `prompt-compare` as a read-only, provider-free comparison of **two current prompt assets**:

```bash
python -m main --no-gui prompt-compare <left-uuid-or-exact-name> <right-uuid-or-exact-name>
python -m main --no-gui prompt-compare <left> <right> --json
```

The command answers a bounded operator question:

> What currently differs between these two prompt assets, and are they directly related through fork lineage?

It complements, rather than replaces:

- `prompt-version-diff` — textual and field comparison of two stored snapshots of the **same** prompt;
- `prompt-lineage` — parent/child relations of **one** prompt;
- future `prompt-dependencies` / `prompt-impact` — dependency and change-risk analysis.

## Confirmed implementation reality

- `_resolve_prompt_reference()` already provides the canonical UUID-or-unique-exact-name resolution contract. Both operands must reuse it independently.
- `Prompt` already exposes current name, description, category, tags, language, context, active state, source, version, quality/usage/rating counters, related prompt IDs, and timestamps.
- `TemplateRenderer.extract_variables()` can derive local Jinja variable names and validate syntax, without rendering or model calls.
- `PromptManager.diff_prompt_versions()` intentionally rejects versions belonging to different prompts. It cannot be reused as the cross-asset comparison API.
- Fork lineage can be read through `get_prompt_parent_fork()` and `list_prompt_forks()`.
- `usage_count`, `rating_count`, `rating_sum`, and `quality_score` are current persisted fields.
- Per-prompt execution analytics exists in `HistoryTracker`, but is not currently exposed as a public `PromptManager` method. V1 must not silently add a parallel analytics seam merely to enrich comparison output.

## V1 contract

### Inputs

- Required positional `left_prompt_id` and `right_prompt_id`.
- Each accepts a UUID or one unique exact name.
- Optional `--json` for structured output.
- Same prompt passed twice is valid but reports no differences; it is useful for automation sanity checks and does not need special semantics.

### Comparison dimensions

1. **Identity and state**
   - ID, name, version, source, active state, last modified timestamp.

2. **Current metadata differences**
   - description, category/category slug, tags, language, scenarios, author, related prompt IDs, favorite state, and selected operational counters.
   - Exclude opaque extension fields (`ext1`–`ext5`) and raw embedding vectors (`ext4`) from v1 to avoid an unclear/unstable contract.

3. **Prompt body**
   - Stable unified text diff of `context`, labelled with the two prompt names/IDs.
   - Do not render the templates and do not call a model.

4. **Template variables**
   - independently derived variables for left/right;
   - `shared`, `left_only`, and `right_only` variable sets;
   - local template parsing errors returned as comparison findings, not hidden.

5. **Direct lineage relation**
   - `left_is_child_of_right`, `right_is_child_of_left`, or `none`.
   - V1 does **not** infer transitive ancestry or a full dependency graph.

6. **Persisted operational counters**
   - usage count, rating count, rating sum, computed average rating when `rating_count > 0`, and quality score.
   - These are reported as facts, not as a quality verdict.

### Text output

Compact sections in this order:

```text
Prompt comparison
Left:  ...
Right: ...

State
...

Metadata differences
...

Template variables
shared: ...
left only: ...
right only: ...

Lineage
relationship: ...

Operational counters
...

Body diff
...
```

Use `(none)` consistently for empty sets/differences. A no-difference comparison must say so explicitly.

### JSON output

Proposed stable top-level shape:

```json
{
  "left": {"id": "...", "name": "..."},
  "right": {"id": "...", "name": "..."},
  "state": {"left": {}, "right": {}},
  "metadata_differences": {},
  "variables": {
    "left": [],
    "right": [],
    "shared": [],
    "left_only": [],
    "right_only": [],
    "errors": []
  },
  "lineage": {"relationship": "none"},
  "operational_counters": {"left": {}, "right": {}},
  "body_diff": ""
}
```

### Exit semantics

- `0`: comparison completed, including comparisons with no differences or template parse findings.
- `4`: either named/UUID prompt is not found.
- `5`: either operand is ambiguous under the existing exact-name contract.
- `7`: repository or lineage read failure.

Template parse errors remain data in the report rather than command failure: the command's job is to compare known stored assets, and `prompt-validate` remains the technical validity gate.

## Explicit anti-scope

- No provider/model calls, model evaluation, semantic similarity, embeddings, or mutation.
- No execution-history aggregates, latency, token/cost, or model comparisons in v1; there is no public manager facade yet for those metrics.
- No version-snapshot selection; use `prompt-version-diff` for same-prompt historical snapshots.
- No dependency graph, chain-consumer analysis, or blast-radius decision; those belong to `prompt-dependencies` and `prompt-impact`.
- No subjective writing advice; that belongs to future `prompt-lint`.
- No raw `ext*` comparison or embedding-vector output.

## Proposed implementation slices

1. **Core comparison report**
   - Add a typed, provider-free module (for example `core/prompt_comparison.py`).
   - Build metadata projection, safe average rating, variable comparison, unified body diff, and deterministic report ordering.
   - Unit-test equal prompts, metadata/body/variable differences, and invalid-template capture.

2. **CLI integration**
   - Add parser and `COMMAND_SPECS` entry.
   - Resolve both prompts through `_resolve_prompt_reference()`.
   - Read direct parent lineage for both prompts and derive only direct relationship.
   - Add text and `--json` renderer.
   - Add entrypoint tests for UUID/name resolution, differences, no differences, and lineage direction.

3. **Docs and verification**
   - Update developer CLI index and changelog only after behavior is delivered.
   - Update the idea backlog to mark the command delivered and keep `prompt-dependencies` / `prompt-impact` as separate successors.

## Verification plan

```bash
.venv/bin/pytest tests/test_prompt_comparison.py tests/test_main_entry.py -q
.venv/bin/ruff check core/prompt_comparison.py cli/parser.py cli/commands.py tests/test_prompt_comparison.py tests/test_main_entry.py
.venv/bin/ruff format --check core/prompt_comparison.py cli/parser.py cli/commands.py tests/test_prompt_comparison.py tests/test_main_entry.py
.venv/bin/pyright core/prompt_comparison.py cli/parser.py tests/test_prompt_comparison.py tests/test_main_entry.py
.venv/bin/python -m main --help
.venv/bin/python -m main prompt-compare --help
git diff --check
```

Then run the full project suite before commit.

## Done criteria

- Both prompt operands follow the existing exact UUID/name contract.
- Identical input assets give deterministic no-difference output.
- Text and JSON expose current asset differences, variables, direct lineage, and persisted counters without model/provider access.
- `prompt-version-diff`, `prompt-validate`, and `prompt-history` retain their separate responsibilities.
- No new persistence schema or public execution-analytics facade is introduced.

## Strongest alternative and decision trigger

**Alternative:** wait and ship `prompt-dependencies` first, then make `prompt-compare` a dependency-aware, impact-rich command.

**Why not now:** a compact current-state comparison can reuse confirmed current seams with no new storage or graph model; dependency/impact semantics still need their own design.

**Change recommendation if:** operators repeatedly need to know which chains/scenarios consume an asset before comparing it, rather than which prompt content/metadata differs. Observable signal: comparison requests consistently end in manual `prompt-lineage`/chain inspection or require answering “what will this change affect?” rather than “how are these two prompts different?”.
