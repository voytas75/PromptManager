# Quick Capture automatic draft metadata — bounded execution

Date: 2026-10-26
Status: locally implemented and offline verified; running GUI/live model not verified.
Authority: `docs/product-ssot.md`, `2026-05-10-product-direction-ssot-next-cycle.md`.
User-selected capture/normalization slice; no roadmap expansion.

## Contract and provider policy

Only Quick Capture Save Draft invokes enrichment, before initial persistence/editor handoff.
The dialog discloses automatic submission of the cleaned body to the configured model before saving; no additional per-field consent prompts are introduced for this specifically requested automatic operation. Normal create/edit retains its existing explicit model consent.
Readiness uses the existing dynamic `manager.llm_available` (configured, not a live health probe).
Name, description and scenarios use manager generators; description disables silent fallback.
Category and tags use a bounded manager method through the existing LiteLLM adapter and configured category-generator model/auth/drop-parameter contract. It requests strict JSON and rejects malformed, empty or non-string values; no local inference/category fallback is permitted. Existing ordinary-editor helpers remain unchanged. Success means all automatically populated fields are model-derived.

Overwrite policy: fill blank title/note only; preserve explicit user title/description and source; merge generated tags after existing tags with exact-string deduplication. Category and scenarios are populated automatically. Body remains the existing deterministic Quick Capture conversion (including pre-existing outer-wrapper cleanup); never run pasted Bash. UUID/draft provenance remain owned by the draft record. Failure/incomplete output keeps the entire deterministic draft, with a visible failure status; no model skips all generators. Cancellation never reaches generation. No real provider calls, settings changes, dependencies, staging, commits or push.

## Evidence

RED: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_quick_capture_metadata.py`
→ 4 failed, missing `generate_quick_capture_metadata`.

GREEN/affected GUI:
`QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_quick_capture_metadata.py tests/test_prompt_editor_flow.py tests/test_quick_capture_dialog.py tests/test_prompt_dialog_refinement.py`
→ 58 passed (6.93s).
New metadata tests use fake manager/generator callbacks and autouse fail-on-network socket connect/create_connection guards. Coverage: generated defaults, manual-field preservation/tag merge, no model and late failure rollback. Flow tests verify callback handoff/status with fake processing indicator.

Nearby smoke:
`QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_draft_promote_dialog.py tests/test_name_generation.py tests/test_scenario_generation.py`
→ 30 passed (8.19s).

`.venv/bin/ruff check .` → all checks passed.
`.venv/bin/pyright gui/prompt_generation_service.py gui/prompt_editor_flow.py gui/dialogs/quick_capture.py tests/test_quick_capture_metadata.py tests/test_prompt_editor_flow.py tests/test_quick_capture_dialog.py`
→ 0 errors, 0 warnings, 0 informations (strict project config; only tool version advisory).
`git diff --check` → clean.

## Limits and preservation

No full-suite/coverage run, live provider probe or running-GUI activation claimed. Category/tags JSON is validated locally, not provider-enforced JSON mode; invalid output rolls back enrichment rather than accepting inferred metadata. Generation consists of sequential helpers under ProcessingIndicator; provider timeout/cancellation behavior is unchanged. The status callback is supplemented by an untimed acknowledged QMessageBox before editor handoff; creation now shares the generation worker.
Preserved pre-existing dirty `docs/STATUS.md` and untracked storage-fault ledger; neither edited by this slice.

## Strict model-only correction evidence

Replaced prior category fallback and heuristic tags after review: the original implementation did not satisfy the all-five-fields requirement.

RED: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_quick_capture_metadata.py` → 5 failed, 2 passed. Two failures exposed the forbidden old category path; three additionally exposed an invalid test comparison of separately minted draft UUID/timestamps, corrected to compare preserved fields.

Final affected/nearby verification:
`QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_quick_capture_metadata.py tests/test_prompt_editor_flow.py tests/test_quick_capture_dialog.py tests/test_prompt_dialog_refinement.py tests/test_draft_promote_dialog.py tests/test_name_generation.py tests/test_scenario_generation.py`
→ 97 passed (5.87s). Metadata tests guard socket connect/create_connection and inject fake adapter completions; prove model category/tags, auth/model/drop-parameter forwarding, invalid output rejection, manual preservation and honest failure rollback.

`.venv/bin/pyright core/prompt_manager/generation.py gui/prompt_generation_service.py gui/prompt_editor_flow.py gui/dialogs/quick_capture.py tests/test_quick_capture_metadata.py tests/test_prompt_editor_flow.py tests/test_quick_capture_dialog.py` → 0 errors, 0 warnings, 0 informations (version advisory only).
`.venv/bin/ruff check .` → all checks passed.
`git diff --check` → clean.
No real provider calls, config/dependency changes, commit or push.

## Six review corrections — final offline evidence

Fixed stale category slug (replace with category/category_slug=None), independent category routing, opt-in strict scenario parsing, registry-compatible category admission plus GUI domain-error catch, initial creation on the same worker, and an untimed acknowledged outcome before the modal editor. Preserve explicit fields/body/source/provenance and atomic enrichment rollback. Real SQLite registry normalization/admission, real scenario completion/parser and rollback, distinct name/category routes, real ProcessingIndicator thread IDs and QMessageBox acknowledgement ordering cover the previously missed seams.

RED, using pre-import offline runner:
- `.venv/bin/python /home/voytas/.hermes/cache/scratch/quick_capture_offline.py -q tests/test_quick_capture_metadata.py` → 19 failed, 7 passed (route lookup, stale slug, invalid category and missing strict parser).
- Same runner `-q tests/test_prompt_editor_flow.py -k acknowledged --tb=short` → 3 failed, 7 deselected (editor opened without acknowledgement).

The scratch runner clears inherited environment, isolates HOME and PROMPT_MANAGER_CONFIG_JSON to scratch, disables dotenv reads and blocks socket connect/connect_ex/create_connection plus DNS getaddrinfo/gethostbyname before project imports. Fake adapter completions only; no live/native acceptance.

Final affected + nearby smoke command:
`.venv/bin/python -m coverage run --rcfile=/dev/null --timid --data-file=/home/voytas/.hermes/cache/scratch/qc-final.coverage /home/voytas/.hermes/cache/scratch/quick_capture_offline.py -q tests/test_quick_capture_metadata.py tests/test_prompt_editor_flow.py tests/test_quick_capture_dialog.py tests/test_prompt_dialog_refinement.py tests/test_draft_promote_dialog.py tests/test_name_generation.py tests/test_scenario_generation.py`
→ **114 passed (5.27s)**. Earlier uninstrumented same selection: 113 passed before the final domain-error regression; final instrumented run includes it.

Coverage export:
`.venv/bin/python -m coverage json --rcfile=/dev/null --data-file=/home/voytas/.hermes/cache/scratch/qc-final.coverage -o /home/voytas/.hermes/cache/scratch/qc-coverage.json --include='core/scenario_generation.py,core/prompt_manager/generation.py,gui/prompt_editor_flow.py,gui/prompt_generation_service.py'`
→ JSON written. Programmatic intersection of executable lines and `git diff --unified=0` added lines (entire dirty slice): generation 32/35 (91.4%), scenario generator 14/15 (93.3%), editor flow 14/14 (100%), service 24/24 (100%). Not whole-module/full-suite coverage. Existing .coveragerc omits scenario generation, hence scratch rc override. pytest-cov C tracing and module-filtered coverage attempts crashed with SIGSEGV; unrestricted timid tracing passed. No configuration changes to work around this.

Final gates:
- `.venv/bin/ruff check .` → all checks passed.
- `.venv/bin/ruff format --check .` → 291 files already formatted.
- `.venv/bin/pyright` → 0 errors/warnings/informations (configured strict gate).
- `.venv/bin/pyright core/scenario_generation.py core/prompt_manager/generation.py gui/prompt_generation_service.py gui/prompt_editor_flow.py gui/dialogs/quick_capture.py tests/test_quick_capture_metadata.py tests/test_prompt_editor_flow.py tests/test_quick_capture_dialog.py` → 0 errors/warnings/informations. Both emit version advisory only.
- `git diff --check` → clean.

No full suite or live/native model/UI activation claimed. Provider timeout/cancellation and existing manager persistence/index compensation remain unchanged. Acknowledgement is intentionally an extra untimed message, not an editor banner. Preexisting docs/STATUS.md and storage-fault ledger untouched; no staging/commit/push.
