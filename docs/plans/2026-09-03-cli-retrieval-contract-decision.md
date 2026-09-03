# PromptManager — CLI Retrieval Contract Decision

**Status:** decided
**Date:** 2026-09-03
**Canonical product SSOT:** `docs/product-ssot.md`

## Decision

`prompt-find` remains a deterministic, read-only catalog filter. It matches the documented text fields and composes with `--category`, `--tag`, `--source`, `--active`, `--limit`, and `--json` without requiring ChromaDB or an embedding provider.

GUI search remains the semantic retrieval surface through `PromptManager.search_prompts(...)` and its explicit search-availability state.

## Reasoning

This is an intentional complementary interface boundary, not a parity defect:

- CLI stays usable for local/offline inspection and automation when semantic retrieval dependencies are unavailable.
- GUI keeps semantic ranking and its provider-dependent failure posture visible to interactive operators.
- Replacing `prompt-find` with semantic retrieval would silently weaken the CLI's deterministic and dependency-light contract.

## Consequences

- Do not modify `prompt-find` to call `search_prompts(...)`.
- Do not describe `prompt-find` as semantic search.
- A future semantic CLI command, if justified by observed operator need, must be separately named and specify its embedding/Chroma requirement, error behavior, filters, and offline posture.

## Verification basis

- `cli/commands.py:run_prompt_find()` reads the catalog and applies deterministic field/filter matching.
- `gui/prompt_list_coordinator.py:fetch_prompts()` calls `manager.search_prompts(...)` for non-empty interactive search.
- `core/prompt_manager/search.py:search_prompts()` depends on an embedding provider and Chroma collection.

## Next selection rule

The next product slice should not reopen CLI retrieval semantics without an explicit operator requirement. Prefer a fresh, bounded capture-quality probe before any new retrieval wording or ranking work.
