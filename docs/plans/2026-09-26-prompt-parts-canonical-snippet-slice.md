# Prompt Parts — canonical snippet slice

Status: implemented (Git delivery state is verified separately)
Product authority: `docs/product-ssot.md`; this note tracks only the bounded implementation.

## Decision and boundary

Keep Prompt Parts as a local library for manual reuse. Make the fragment's own text durable and unambiguous before considering CLI or reference-based composition. No automatic composition, new CLI, immutable revision history, or provider calls in this slice.

## Delivered

- `ResponseStyle.snippet` is the canonical fragment text, independent of description, optional format instructions, and examples. SQLite has a corresponding field; create/edit/reopen round-trips preserve it. A blank description remains blank rather than being filled with the snippet.
- When opening a legacy database without the field, copy existing format instructions (if nonblank), otherwise description. Preserve the old fields as-is; the migration runs only when the field is missing. Previously discarded snippet text cannot be recovered.
- GUI detail/copy and plain-text export include the snippet. Newly created parts no longer silently duplicate it into description, optional format instructions, or examples.
- Developer documentation and changelog now reflect the actual manual-reuse boundary. The product SSOT remains unchanged because the product direction did not change.

## Evidence

- RED: focused repository and GUI regressions failed because `ResponseStyle` had no `snippet`; a second RED showed implicit duplication into optional fields.
- GREEN: focused `QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_response_styles.py tests/test_prompt_parts_dialog.py tests/test_repository_branches.py tests/test_prompt_manager_storage.py -q` — 49 passed before the final blank-description regression; the subsequent full suite includes that test.
- Full suite after the final code change: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q` — 1038 passed, 1 skipped.
- `.venv/bin/ruff check .`, changed-file `ruff format --check`, full `.venv/bin/pyright`, and `git diff --check` — passed.
- Live provider calls: none. Host GUI acceptance beyond Qt offscreen: do weryfikacji.

## Closure

This implementation slice is complete. The separate CLI v1 delivery is tracked in `docs/plans/2026-09-26-prompt-parts-cli-v1.md`; that plan does not alter this slice's original boundary. Structured export and version-pinned references remain separate product decisions.
