# PromptManager — Potential future CLI options

**Status:** idea backlog / not approved for implementation
**Owner:** PromptManager Team
**Source:** operator product review, retained for future prioritization

## Architectural observation

PromptManager already manages prompt assets well through CRUD, versions, lineage, retrieval, execution history, chains, and diagnostics. The strongest future gap is treating prompts more like software with a disciplined lifecycle:

> lint → validate → test → evaluate → impact analysis → controlled release

This is an idea backlog, not an active roadmap. Each candidate needs code/test recon and separate scope approval before implementation.

## Priority candidates

1. **`prompt-test <uuid|name>`**
   Run a named or default provider-backed/controlled test suite against a prompt.
   - Candidate follow-ups: `prompt-test-suite-list`, `prompt-test-suite-show`, `prompt-test-suite-run`.
   - Desired output: totals, passed/failed cases, and specific failed case IDs/reasons.
   - Boundary: distinguish deterministic/provider-free checks from explicitly authorized model evaluation.

2. **`catalog-check`** — delivered in `docs/plans/2026-09-21-catalog-check-v1.md`
   Whole-catalog integrity pass analogous to lint/check tooling.
   - v1 covers exact duplicate names and normalized bodies, invalid template syntax, broken related-prompt references, missing stored embeddings, and chains referencing missing prompts.
   - v1 is read-only, deterministic, provider-free, and exposes text plus `--json` output; broader candidates remain deferred.

3. **`prompt-impact <uuid|name>`**
   Change-risk view before editing a prompt.
   - Candidate evidence: chains using it, scenarios, fork descendants, latest executions, usage count, models used, and related dependent assets.
   - This should reuse dependency/lineage data, not invent a separate graph model.

4. **`prompt-validate <uuid|name>`**
   Technical, provider-free prompt validation without supplying template values or calling a model.
   - Candidate checks: required fields, metadata shape, template syntax, referenced versus declared variables, and model/type compatibility where a real contract exists.
   - It should extend/reuse `prompt-render` validation primitives rather than duplicate parsing.

5. **`prompt-evaluate <uuid|name>`**
   Dataset/history-based quality evaluation, distinct from syntax validation.
   - Candidate metrics: pass rate, latency, tokens/cost, evaluator scores, and comparison with a previous version.
   - Provider/dataset runs require an explicit acceptance boundary and reproducible evidence contract.

6. **`prompt-dependencies <uuid|name>`**
   Direct and reverse dependency view.
   - Candidate fields: variables, chains using the prompt, fork source, scenario/reference links.
   - Candidate option: `--reverse` for consumers of the asset.

## Other candidates

### Lifecycle and metadata

- **`prompt-deprecate <id> [--reason TEXT]`** and **`prompt-activate <id>`**
  - Surface the existing active/inactive lifecycle state as explicit operations.
  - Verify semantics and existing GUI/state model first.

- **`prompt-tag <id> add|remove <tag>`**, **`tag-list`**, **`tag-show <tag>`**
  - Convenience metadata operations; lower priority than quality/integrity tooling.

- **`prompt-clone <id> --name NAME`**
  - Only if product semantics genuinely differ from the existing `prompt-fork` lineage-preserving path.
  - Default: do not add until that distinction is proven useful.

### Comparison and maintenance

- **`prompt-compare <id1> <id2>`**
  - Operational/semantic comparison of metadata, variables, usage, models, evaluation, and performance; complementary to text-level `prompt-version-diff`.

- **`prompt-gc --dry-run`**
  - Detect orphaned embeddings, stale artifacts, dangling lineage references, orphaned records, unused scenarios, and unused chains.
  - Any destructive cleanup must remain separately confirmed and start with dry-run output.

- **`workspace-export PATH` / `workspace-import PATH`**
  - Potential backup/migration of prompts, chains, scenarios, metadata, lineage, and later test suites.
  - Candidate options: `--dry-run`, `--no-overwrite`, `--include-history`, `--include-embeddings`.
  - Derived embeddings should remain excluded by default.

### Prompt engineering guidance

- **`prompt-lint <uuid|name>`**
  - Design-quality warnings rather than technical validity.
  - Candidate diagnostics: vague instructions, redundant/conflicting constraints, weakly used variables, and length deltas versus prior versions.
  - Keep `validate` and `lint` separate:
    - `validate`: technically correct and executable contract.
    - `lint`: likely design/maintainability issues.

## Candidate delivery order

Default sequence from the review:

1. `prompt-test`
2. `catalog-check`
3. `prompt-impact`
4. `prompt-validate`
5. `prompt-evaluate`
6. `prompt-dependencies`

## Product and scope guardrails

- Keep PromptManager asset-first; operational quality tooling must support reuse/refine decisions, not turn the product into a broad workflow platform.
- Prefer read-only, deterministic, provider-free checks first.
- Separate deterministic validation from model-backed evaluation at the CLI, data, test, and authorization boundaries.
- Reuse existing version, lineage, history, chain, renderer, and repository seams before adding data models.
- Treat destructive operations (`gc`, import overwrite) as dry-run-first and explicitly confirmed.
- Do not schedule these candidates automatically; select one only after implementation-reality review proves an unmet seam.
