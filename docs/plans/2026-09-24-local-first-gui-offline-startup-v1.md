# PromptManager — local-first GUI startup without LiteLLM v1

Status: implemented and verified locally (delivery: commit and push requested)
Owner: Wojtek / Prompt Manager Team
Product SSOT: `docs/product-ssot.md`
Near-term plan: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Problem and decision

The default entrypoint refused to start the GUI when the fast LiteLLM model or API key was absent, despite the local-first product contract and working provider-free catalog commands. Separate catalog/service failures must still fail normally. Model availability must remain visible in diagnostics and enforced at the execution seam, not at application entry.

## Bounded scope

- Remove only the entrypoint's model/key startup gate and its unused helper.
- Keep settings validation, manager/service initialization, offline availability state, model execution guards, embeddings routing, provider credentials and persistence unchanged.
- Add entrypoint regressions for model-free GUI/headless startup and a real, isolated offscreen GUI operator-path test.
- Explain the required deterministic embedding selection for a fully provider-free installation.

## Acceptance and evidence

- Entrypoint without model/key initializes local services and launches the GUI; the no-GUI mode also starts. Focused tests pass.
- Real `main.main()` in offscreen Qt with temporary JSON settings, SQLite, ChromaDB, home and XDG paths: Quick Capture creates a draft, the detail selects it, Copy Prompt copies its body, Open in Workspace populates the editor, and Run Prompt refuses while `llm_available=False` and no executor exists. The test asserts the model execution method is never called.
- Invalid settings and manager initialization failure continue to return nonzero; `--print-settings` still reports the missing model/key. Focused entrypoint/diagnostics tests pass.
- Full provider-free gate: **898 passed, 1 skipped**, core coverage **81.68%**. `ruff check .`, `ruff format --check .`, full configured strict `pyright`, `uv lock --check`, and `git diff --check` pass.

## Limits

- No live provider, native Windows GUI acceptance, or release in this slice.
- The example config selects LiteLLM embeddings; fully provider-free search needs an explicit deterministic embedding backend. This slice does not change embedding defaults or investigate semantic retrieval quality.
- Existing GUI/CLI diagnostics classify model/key absence as `FAIL` for model capabilities; they are retained as such and no longer block local startup.
