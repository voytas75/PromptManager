# PromptManager — Potential future CLI options

**Status:** living idea backlog / not approved for implementation
**Owner:** PromptManager Team
**Source:** operator product review, updated after delivery of `catalog-check`

## Current baseline

PromptManager already covers CRUD, import/export, versioning, lineage, retrieval, execution history, chains, diagnostics, random selection, and provider-free catalog integrity checks.

Delivered quality/integrity tooling:

- **`catalog-check [--json]`** — delivered in [`2026-09-21-catalog-check-v1.md`](2026-09-21-catalog-check-v1.md).
  - Detects duplicate exact names and normalized bodies, invalid Jinja syntax, broken related-prompt links, missing stored embeddings, and chain steps targeting missing prompts.
  - Is deterministic, read-only, provider-free, and returns nonzero only for error-severity findings.

## Architectural direction

The remaining opportunity is to treat prompts more like software with a disciplined lifecycle:

> lint → validate → test → evaluate → impact analysis → controlled release

This is an idea backlog, not an active roadmap. Each candidate requires code/test reconnaissance and separate scope approval before implementation.

## Next priority candidates

1. **`prompt-validate <uuid|name>`** — delivered in [`2026-09-21-prompt-validate-v1.md`](2026-09-21-prompt-validate-v1.md)
   Technical, provider-free validation of a single persisted prompt without supplying template values or calling a model.
   - v1 validates blank name/description, empty body, Jinja syntax, local related-prompt references, and blank/duplicate tags; it reports detected template variables as information.
   - It reuses the UUID-or-unique-exact-name lookup and Jinja parser. Declared-variable schemas and model/type compatibility remain deferred because no durable prompt contract exists.

2. **`prompt-dependencies <uuid|name> [--reverse]`**
   Direct and reverse dependency view for an individual prompt.
   - Candidate fields: variables, consuming chains, fork source/descendants, scenarios and reference links.
   - Forms the evidence base for `prompt-impact`; reuse lineage/repository seams rather than creating a parallel graph model.

3. **`prompt-impact <uuid|name>`**
   Change-risk view before editing a prompt.
   - Candidate evidence: consuming chains, scenarios, fork descendants, latest executions, usage count, models used, and dependent assets.
   - Depends on the dependency view being well-defined; no independent graph model.

4. **`prompt-test <uuid|name>`**
   Run a named or default regression suite against a prompt.
   - Candidate follow-ups: `prompt-test-suite-list`, `prompt-test-suite-show`, `prompt-test-suite-run`.
   - Desired output: totals, passed/failed cases, and precise failed case IDs/reasons.
   - First decide the durable test-case and expected-result contract; distinguish deterministic tests from explicitly authorised provider runs.

5. **`prompt-evaluate <uuid|name>`**
   Dataset/history-based response-quality evaluation, separate from validation and regression tests.
   - Candidate metrics: pass rate, latency, token/cost data, evaluator scores, and comparison to a previous version.
   - Provider/dataset runs need an explicit acceptance boundary and reproducible evidence contract.

6. **`prompt-lint <uuid|name>`**
   Design-quality guidance, distinct from technical correctness.
   - Candidate diagnostics: vague instructions, redundant/conflicting constraints, weakly used variables, and length deltas versus prior versions.
   - Keep the boundary explicit:
     - `validate`: technically correct and executable contract;
     - `lint`: likely design/maintainability problems.

## Lower-priority candidates

### Lifecycle and metadata

- **`prompt-deprecate <id> [--reason TEXT]`** and **`prompt-activate <id>`**
  - Surface active/inactive lifecycle state as explicit operations.
  - Verify the existing GUI/state semantics before adding write paths.

- **`prompt-tag <id> add|remove <tag>`**, **`tag-list`**, **`tag-show <tag>`**
  - Convenience metadata operations; lower priority than quality, integrity, and change-safety tooling.

- **`prompt-clone <id> --name NAME`**
  - Add only if product semantics differ meaningfully from the existing lineage-preserving `prompt-fork`.
  - Default: do not add until the distinction is proven useful.

### Comparison and maintenance

- **`prompt-compare <id1> <id2>`**
  - Operational comparison of metadata, variables, usage, models, evaluations, and performance; complements text-level `prompt-version-diff`.

- **`prompt-gc --dry-run`**
  - Detect orphaned embeddings, stale artifacts, dangling lineage references, orphaned records, unused scenarios, and unused chains.
  - Any cleanup remains separately confirmed and dry-run-first.

- **`workspace-export PATH` / `workspace-import PATH`**
  - Potential backup/migration of prompts, chains, scenarios, metadata, lineage, and later test suites.
  - Candidate options: `--dry-run`, `--no-overwrite`, `--include-history`, `--include-embeddings`.
  - Derived embeddings remain excluded by default.

## Recommended delivery sequence

1. `prompt-dependencies`
2. `prompt-impact`
3. deterministic `prompt-test` foundation
4. `prompt-evaluate` — only after an approved provider/dataset evidence contract
5. `prompt-lint`

**Counterfactual:** advance `prompt-test` before dependencies/impact if users already have stable, valuable test datasets and can name an acceptance metric. Observable signal: repeated manual evaluation of the same prompt inputs or provider runs being used as a de facto regression gate.

## Product and scope guardrails

- Keep PromptManager asset-first; quality tooling should support reuse/refine decisions, not turn it into a broad workflow platform.
- Prefer read-only, deterministic, provider-free checks first.
- Separate deterministic validation from model-backed evaluation at CLI, data, test, and authorisation boundaries.
- Reuse existing version, lineage, history, chain, renderer, and repository seams before adding data models.
- Treat destructive operations (`gc`, import overwrite) as dry-run-first and explicitly confirmed.
- Do not schedule candidates automatically; select one only after an implementation-reality review proves an unmet seam.
