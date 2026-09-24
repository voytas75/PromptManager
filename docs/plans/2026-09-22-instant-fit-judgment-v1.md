# PromptManager — Instant Fit Judgment v1

Status: completed (2026-09-22; historical Stage A execution ledger, not an active next slice)
Owner: Wojtek / Prompt Manager Team
Updated: 2026-09-22
Canonical product SSOT: `docs/product-ssot.md`
Canonical near-term plan: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Product decision

PromptManager remains the local-first canonical home for prompt assets. The next cycle removes friction from finding, understanding, and safely reusing those assets; it does not broaden the product into a general AI workspace.

Every proposed slice must answer:

> Does this make an operator use their prompts more often and with more confidence?

If not, it does not enter the cycle.

## Approved sequence

1. **Stage A — Instant Fit Judgment**
   - Show one compact, truthful evidence line in GUI result rows and detail, and in CLI `prompt-find` / text `prompt-show`.
   - Use only existing local evidence: usage count, persisted rating aggregates, and last recorded run when that surface already has it.
   - Keep JSON records structurally unchanged; do not introduce a synthetic score, ranking change, provider call, persistence change, or dashboard.

2. **Stage B — Intentional capture continuity**
   - Start with explicit capture from clipboard with preview into a draft.
   - Consider an application hotkey only after real use validates it.
   - Do not add a background clipboard watcher without a separate privacy and platform decision.

3. **Stage C — Baseline / variant learning loop**
   - Make baseline versus variant and side-by-side evidence easy to read.
   - Add only operator-controlled actions around existing run evidence; do not claim that a version is better without comparable evidence.

4. **Stage D — Local portability and calm trust**
   - Add local-first Markdown + JSON export/import only when the asset loop needs it.
   - Improve status explanations only where they unblock the asset loop; optional model availability is not catalog failure.

## Stage A contract

### GUI

- Result rows expose one muted fit cue derived only from persisted prompt aggregates.
- Detail exposes one fit summary that can also include the latest local execution timestamp.
- No evidence is represented honestly as `No run evidence yet`; existing decision and next-action cues remain responsible for reuse guidance.

### CLI

- `prompt-find` text rows include the same aggregate fit cue.
- Text `prompt-show` includes a compact `Fit` section and may add the latest local execution timestamp.
- `--json` payloads remain clean JSON and retain their current record shapes.

### Verification

- Red tests first for aggregate evidence, no-evidence behavior, GUI list/detail rendering, and CLI text rendering.
- Provider-free focused tests, then full provider-free suite and normal repository gates.
- Use deterministic local fixtures; no live model or embedding-provider calls.

## Explicitly out of scope

- New ranking models, confidence scores, persistence/schema migrations, or analytics dashboards.
- Background monitoring, global OS hotkeys, multi-user collaboration, chain expansion, and external sharing.
- Provider configuration or live execution.

## Completion record

Completed 2026-09-22:

- Added one shared, provider-free evidence formatter for persisted usage and rating aggregates; absent or incomplete evidence stays explicit rather than becoming a synthetic score.
- Added the compact cue to the existing GUI result delegate and detail view, and to text-mode `prompt-find` and `prompt-show`; structured JSON contracts are unchanged.
- Corrected the GUI history controller to use the production `list_executions_for_prompt()` API, restoring its existing last-run/decision evidence path outside test stubs.
- Verified with the full provider-free suite (`885 passed, 1 skipped`, core coverage `81.68%`), Ruff lint/format, CI-scope Pyright, `uv lock --check`, isolated local CLI smoke, and `git diff --check`.

Deferred by design: last-run status/timestamp is already rendered in the existing detail run-summary path; no new retrieval ranking, confidence score, persistence, provider path, dashboard, or global capture behavior was added.
