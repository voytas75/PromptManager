# PromptManager — GUI/CLI logical tag parity v1

Status: implemented and verified; delivery tracked by Git history
Product SSOT: `docs/product-ssot.md`

## Confirmed discrepancy

- `Promote Draft` writes the complete comma-separated tag list, preserving spelling (including `Ops` and `ops` within one prompt); CLI `prompt-tag add|remove` edits one tag and compares case-insensitively.
- CLI `tag-list` aggregates logical tags case-insensitively and `tag-show` matches case-insensitively; the GUI filter previously displayed both case spellings and narrowed on exact spelling. The detail view accurately displayed stored metadata.
- The mismatch was reproduced on an isolated SQLite catalog and offscreen GUI. A two-prompt check (`Ops` / `ops`) showed one CLI logical tag with two prompts but two disjoint GUI filters.

## Bounded implementation

- Reuse `core.prompt_tagging.build_tag_catalog` for GUI dropdown labels, sorted alphabetically, and match selected tag names by trimmed casefolded membership.
- Retain the current logical filter when the available display spelling changes (including pending tag selection); do not rename or migrate stored tags, change the detail view, or modify CLI commands.
- Clarify comma-separated full-list replacement at promotion versus single-tag CLI mutation in `docs/canonical-usage-path-v1.md`.

## Acceptance

- One GUI dropdown option for each case-insensitive logical tag.
- Selecting the logical tag finds all prompts regardless of stored spelling and survives a refresh that changes the display spelling.
- Existing CLI catalogue, prompt tag mutation, promotion, and other GUI filters retain their behavior.
- Provider-free focused tests, full tests, Ruff, strict Pyright, lock check, and isolated offscreen runtime check pass. No user database or provider is touched.

## Verification

- RED: two new GUI tests failed on duplicate `Ops`/`ops` options and lost selection after spelling change.
- GREEN: focused coordinator, filter panel, tag helper, and promote-dialog pack: `38 passed`.
- Full provider-free suite: `900 passed, 1 skipped`; core coverage `81.68%` (80% gate). `ruff check .`, `ruff format --check .`, full configured strict `pyright` (0 errors), `uv lock --check`, and `git diff --check` passed.
- Repeated isolated offscreen GUI/CLI smoke with temporary SQLite/ChromaDB and deterministic embeddings: promoted `Ops, ops, Review`; CLI grouped `Ops` once and idempotent `add OPS` reported `changed: false`; `add CI` persisted; GUI detail retained stored spelling `[Ops, ops, Review, CI]`; filter options were `[All tags, CI, Ops, Review]`, and selecting `Ops` matched the prompt. No paid provider or user database was used.
- Two-prompt `Ops`/`ops` selection parity and preservation after refresh are enforced by the new focused tests. Native Windows acceptance remains to verify.
