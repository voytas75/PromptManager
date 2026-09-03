# PromptManager — Update Chroma Rollback Integrity

**Status:** verified delivery record
**Date:** 2026-09-03
**Product priority:** Priority 1 — trustworthy prompt-asset retrieval and reuse
**Canonical product SSOT:** `docs/product-ssot.md`

## Purpose

Record the bounded consistency repair for updating a prompt when SQLite accepts the new asset before the derived Chroma semantic index rejects it.

## Confirmed baseline

- `PromptLifecycleMixin.update_prompt()` writes the updated prompt to SQLite before persisting a synchronously generated Chroma embedding.
- Before this slice, a `PromptStorageError` from the Chroma upsert left SQLite with the new prompt body while semantic retrieval still had the prior record.
- This is a consistency risk on the retrieve → inspect → reuse loop, not a retrieval-ranking change.

## Delivered contract

When synchronous Chroma embedding persistence fails during `update_prompt()`:

1. restore SQLite from the captured `previous_prompt`;
2. re-raise the original `PromptStorageError` so the caller sees the failed update;
3. log a rollback failure without hiding the original Chroma error.

The asynchronous no-embedding path is unchanged.

## Scope

- `core/prompt_manager/lifecycle.py`
- `tests/test_prompt_manager_branches.py`

Out of scope:

- ChromaDB dependency or server configuration;
- retrieval ranking, embedding generation, or retry policy;
- a durable repair queue or reconciliation worker;
- new UI states, panels, or product workflow.

## TDD evidence

- RED in disposable worktree: a failing Chroma upsert raised `PromptStorageError`, but the recording SQLite repository retained the changed body.
- GREEN: the same failure restores the original stored body and still raises `PromptStorageError`.

## Verification

- `pytest tests/test_prompt_manager_branches.py::test_update_prompt_rolls_back_sqlite_when_chroma_upsert_fails tests/test_prompt_manager_branches.py::test_update_prompt_handles_chroma_and_cache_failures -q` → `2 passed`
- `pytest tests/test_prompt_manager_branches.py tests/test_prompt_manager_storage.py -q` → `55 passed`
- `pyright core/prompt_manager/lifecycle.py` → `0 errors`
- Ruff check and formatter check for the two changed paths → passed.
- Final approved code scope: `27 additions + 1 deletion = 28 gross lines`.

## Closure

This closes the reproduced synchronous update inconsistency between SQLite and Chroma. It complements the earlier delete-path ordering contract: neither an unsuccessful derived-index mutation nor a failed derived-index delete silently leaves an asset in a misleading partial state.

**Next selection rule:** do not broaden this into a reconciliation subsystem without a reproduced rollback-failure or delayed-indexing operator symptom. Prefer a fresh read-only asset-loop probe.
