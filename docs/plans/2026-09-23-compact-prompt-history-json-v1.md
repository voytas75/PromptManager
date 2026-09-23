# PromptManager — Compact `prompt-history --json` v1

Status: completed
Owner: PromptManager Team

## Goal

Extend the established bounded prompt-record JSON contract to execution-history inspection:

```bash
python -m main --no-gui prompt-history <uuid-or-name> --json
python -m main --no-gui prompt-history <uuid-or-name> --json --full
```

## Confirmed baseline

- `prompt-show --json` and `prompt-find --json` emit a compact prompt record by default and expose the complete record only via `--json --full`.
- `prompt-history --json` previously embedded `prompt.to_record()` directly under `prompt`, exposing `ext4`.
- `prompt-history` execution entries are a separate history contract and must remain unchanged.

## Decision

Reuse the existing CLI-local `_prompt_json_payload()` renderer for the nested `prompt` field. Add `--full` to `prompt-history`, valid only with `--json`.

## Scope

- `cli/parser.py`: add `prompt-history --full` and extend the existing parser guard.
- `cli/commands.py`: render the nested prompt via `_prompt_json_payload()`.
- Tests, docs, patch version and lock metadata, and this execution ledger.

## Out of scope

- Execution payload shape, history filtering/windowing, ranking, persistence, embeddings, providers, GUI, `Prompt.to_record()`, commit, tag, push, and publish.

## Acceptance criteria

1. `prompt-history --json` omits nested `prompt.ext4` and reports embedding metadata.
2. `prompt-history --json --full` includes nested `prompt.ext4`.
3. `prompt-history --full` fails parser validation.
4. Existing execution JSON remains byte-for-byte equivalent aside from the compact/full `prompt` record.
5. Focused and full provider-free gates pass.

## Completion update

Completed 2026-09-23:

- Added `prompt-history --full`, valid only with `--json`.
- Reused `_prompt_json_payload()` for the nested `prompt` field, aligning `prompt-history` with `prompt-show` and `prompt-find`.
- Default history JSON omits nested `prompt.ext4` and reports bounded embedding metadata; full JSON retains the exact vector.
- Left `analytics` and `executions` payloads unchanged.
- Advanced package and lock metadata from `0.23.2` to `0.23.3`; updated the developer CLI index, changelog, and status.

Verification completed:

- focused history compact/full/parser regressions: 3 passed;
- complete `tests/test_main_entry.py`: 111 passed;
- full provider-free suite: 894 passed, 1 skipped;
- `ruff check .`, `ruff format --check .`, full configured strict Pyright, `uv lock --check`, and `git diff --check`: passed;
- real repository CLI smoke with a stored 3,072-dimensional embedding: compact history JSON omitted `ext4` and reported 3,072 dimensions, full history JSON returned the vector, analytics and executions were equal between forms, and invalid `--full` exited 2 with the expected message.

Unchanged: history filtering/windowing, execution payloads, ranking, persistence, embedding generation/storage, providers, GUI, `Prompt.to_record()`, commit, tag, push, and publish.