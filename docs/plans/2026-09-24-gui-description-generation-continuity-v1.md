# GUI description generation continuity v1

Status: locally verified; remote delivery and exact-SHA CI to verify
Owner: PromptManager Team
Canonical near-term plan: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Reproduced hesitation

The Create Prompt dialog has an editable Description field and a wired description generator, but no visible control to request it. Saving an empty description used to generate it implicitly; with a configured model, the save action could contact a provider without an explicit choice. The same implicit model call existed for name suggestion on body edits and for a blank name on save. A provider-free Qt probe confirmed the description stays blank until save even though it then fills. The CLI `prompt-add` parser independently requires `--description` in inline mode and rejects a JSON entry without description; this is a distinct command contract, not evidence that GUI generation works there.

## Bounded GUI decision

- Provide an explicit Generate description button beside the editable field. With no configured LLM it produces a deterministic local excerpt; with a model configured, it requires confirmation before sending the body to the provider, with cost disclosure and a safe default of No. A refusal does not overwrite manual text. Do not call any real provider in this slice's verification.
- Saving a prompt with an empty description creates a local, editable excerpt from its body only; never call the model implicitly. This includes editing a saved prompt after clearing its description. If no body exists, retain the existing missing-fields validation. Preserve a supplied manual description.
- Defer the automatic name suggestion until save so the first typed character cannot become the name. Blank name at save uses the complete body locally; the existing explicit name Generate button confirms before a configured model call.
- Keep `prompt-add` unchanged for this bounded GUI slice. Decide any optional CLI description-generation flag separately, including its batch/file semantics and dry-run effects. Do not change imports, model routing, persistence schemas, or provider configuration here.

## Evidence

- Existing Qt probe: no description button and no description before save; injected generator called during save. CLI inline `prompt-add --name Demo --prompt-text ... --dry-run` rejects missing `--description` with exit 2, before any import or provider.
- RED GUI tests: missing Generate description button and implicit description generator call during save; an additional RED test exposed implicit provider-backed name generation while typing/saving.
- GREEN focused GUI/flow/offline-startup tests after name and edit fixes: 21 passed; real offline entrypoint saves a prompt with explicit local description generation and preserves Quick Capture and Recent tests. Full local suite: 987 passed, 1 skipped, core coverage 81.66%; repo-wide Ruff lint/format, full strict Pyright, `uv lock --check`, `git diff --check`, and scratch wheel build passed.
- Independent read-only review found the first-character name blocker (`R` for an incrementally typed body). A RED typing-sequence regression confirmed it; the suggestion is now deferred until save. An edit-path test confirms clearing a saved description leads to local replacement without mutating the original prompt. Replacement read-only review found no blocker in this scoped change; focused review suite: 22 passed. No live provider call.
- Boundary: name/description generation on Save is provider-free, but the existing `create_prompt`/update lifecycle may call a configured provider for category insight and embeddings. This slice does **not** promise provider-free persistence overall; the offline integration test uses deterministic embeddings.

## Next decision

Locally verified (987 passed, 1 skipped; core coverage 81.66%; Ruff, strict Pyright, lock, diff check, and wheel). Independent replacement review found no blocker. Only name/description generation on Save is provider-free; existing category/embedding persistence may still contact a configured provider. Publication was authorized; exact-SHA remote delivery/CI remain to verify. CLI generation remains a separate decision, not silently added to `prompt-add`.
