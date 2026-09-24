# PromptManager — `doctor` CLI contract and migration v1

Status: offline stages 1–5 and opt-in `doctor embeddings|analytics --live` delivered; live backend acceptance and exact-SHA CI verified
Owner: PromptManager Team
Product SSOT: `docs/product-ssot.md`
Near-term priority: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md` (supporting trust surface, not a new product center)

## Decision and confirmed baseline

Give the operator one starting command, `prompt-manager doctor`, which answers **what works now, what does not, and the next action**. Keep targeted inspection under `doctor ...` and preserve existing commands as working compatibility entrypoints. Do not collapse asset validation, health, analytics, and repair into one opaque pass.

Today `--print-settings` displays configuration diagnostics but returns 0 on a diagnostic `FAIL`; `catalog-check --json` runs a deterministic integrity report but its CLI path builds `PromptManager`; `diagnostics embeddings` calls `provider.embed(sample_text)`; `diagnostics analytics` is a report/export, not a health check. `main.py` can offer to create missing config at a TTY. `PromptRepository.__init__` ensures a directory/schema, and `build_chroma_client` creates the Chroma directory/collection. Therefore, reusing normal startup/manager construction for default `doctor` would violate the no-mutation/no-provider contract.

## Public command and safety contract

```text
prompt-manager doctor [--json]
prompt-manager doctor config [--details] [--json]
prompt-manager doctor catalog [--json]
prompt-manager doctor index [--json]
prompt-manager doctor embeddings [--json] [--live]
prompt-manager doctor prompt <uuid-or-exact-name> validate [--json]
prompt-manager doctor prompt <uuid-or-exact-name> lint [--json]
prompt-manager doctor prompt <uuid-or-exact-name> test --suite <file> [--json]
prompt-manager doctor chain <definition-file> validate [--json]
prompt-manager doctor analytics [report options] [--live] [--export-csv PATH]
```

- Bare `doctor` is a fast, bounded, **read-only and provider-free** assessment. It reports independent `config`, `local_catalog`, `search_embeddings`, and `model_execution` capabilities with `OK` / `WARN` / `FAIL` / `SKIP` (not assessed). It lists only actionable findings and a short recommended next step; verbose detail belongs to a subcommand. Offline LiteLLM means `model_execution: WARN` while a usable local catalog remains usable; never turn optional TTS, Redis, inference routing, or absent LLM credentials into a false catalog failure. Do not claim successful search or model access without a real probe: distinguish configured/readiness from reachability.
- Bare `doctor` inspects settings and existing paths directly, without `build_prompt_manager`, `PromptRepository(...)`, Chroma `PersistentClient`/`get_or_create_collection`, SQLite schema creation, network, seed data, interactive prompts, or automatic repair. If a config path is explicitly supplied but missing/invalid, report a bounded config `FAIL` without offering to create it. If the default config is absent and defaults suffice, report that fact without creating it. Inspect an existing SQLite DB using a read-only URI; check only bounded structural/readability signals. A missing DB on first-run is not silently created or called corrupt. If the vector store cannot be inspected without writes, mark it `SKIP`/unverified, not `OK`.
- `doctor config` uses the **same effective settings precedence** as application startup but has a separate health classification by capability. `--details` reports a bounded source classification and availability flags; it intentionally omits raw paths, model identifiers, DSNs, and source values because those strings may contain secrets. Do not copy unredacted `--print-settings` output into JSON. Never emit credential prefixes/suffixes or exception text. A credential presence flag is not a network validation.
- `doctor catalog` reuses pure `core.catalog_check.run_catalog_check` but loads prompts/chains through an explicitly read-only repository path. Its issue codes and semantic findings stay unchanged; no manager creation. It may report vector-index inspection as `SKIP` until a proven read-only Chroma path exists. It should distinguish an empty/missing first-run catalog from a corrupt/unreadable existing one. Do not claim storage consistency from equal counts alone; any eventual index check must compare exact identifier sets.
- `doctor index` is an on-demand, provider-free and read-only metadata comparison, never part of bare `doctor`. It uses immutable SQLite readers for the existing catalog and the known Chroma 1.5 `prompt_manager` collection's METADATA segment. It compares exact IDs of catalog prompts with saved `ext4` vectors to Chroma metadata IDs, returns counts only, and fails closed for pending WAL/journals, missing required tables/columns or ambiguous collection/segment. Matching IDs are a best-effort observation without a shared catalog/index snapshot; close writers and repeat a mismatch before repair. HNSW files, vector content/dimensions, search and retrieval quality remain unverified. No Chroma client, repair or provider call; missing first-run files warn without creation.
- `doctor embeddings` without `--live` checks effective configuration only and marks backend/vector reachability unprobed; it does not generate embeddings or inspect the vector index. Explicit `--live` sends one synthetic text to the configured LiteLLM embedding backend (15-second request timeout, no intentional retries), and may call a paid/remote provider as stated in help. It reports only vector usability/dimension, never response text, endpoint or credentials; the vector index is not inspected or written. Preserve the existing `diagnostics embeddings` legacy behavior separately. A future `--fix` is out of scope and requires its own risk decision.
- `doctor prompt ...` and `doctor chain ...` remain **on-demand checks**, never part of bare `doctor`. Reuse pure validators/test/lint functions and a read-only loader where persisted prompt data is needed; no provider calls. `doctor analytics` is explicitly marked `report (not health)`; no part of it influences bare `doctor`. `core.analytics_dashboard.build_analytics_snapshot` currently invokes `manager.diagnose_embeddings()` → `provider.embed(...)`, so default `doctor analytics` must bypass that path (or split out a safe analytics-only snapshot); `--live` opts into the same single synthetic, bounded embedding-backend probe as `doctor embeddings --live`, only after a readable **existing** local count report (and any explicit CSV export) succeeds; missing first-run DB skips the probe. Probe failure keeps the local counts but exits 1 with a sanitized status. This does not validate analytics correctness, the vector index, or retrieval quality. CSV export writes only when `--export-csv PATH` is explicitly requested, and help must state this exception; a CSV may exist when a later live probe fails.
- `doctor --help`, family help, and leaf help exit 0 and explain effects, network/cost flags, output and next command. Keep `--no-gui` unnecessary for `doctor`; both `prompt-manager` and `python -m main` expose the same parser path.

## Output and exit contract

Default text example (illustrative contract, **not a measured runtime result**):

```text
Doctor: local catalog usable; model execution unavailable
OK   Local catalog — existing SQLite readable
WARN Search/embeddings — backend configured, reachability not tested
WARN Model execution — LiteLLM credentials missing
Next: add LiteLLM credentials only if you need to run prompts.
Details: prompt-manager doctor config --details
```

`--json` emits exactly one bounded result on **stdout**, including for a diagnosed unhealthy state; no startup banners, log noise, credentials, full prompts, or raw provider exception text. Shape v1:

```json
{
  "schema_version": 1,
  "command": "doctor",
  "ok": true,
  "status": "WARN",
  "checks": [
    {"id": "local_catalog", "status": "OK", "code": "DB_READABLE", "message": "Existing SQLite catalog readable", "next_step": null},
    {"id": "model_execution", "status": "WARN", "code": "LLM_NOT_CONFIGURED", "message": "Model execution unavailable", "next_step": "Configure LiteLLM only if you need model runs"}
  ],
  "next_step": "Configure LiteLLM only if you need model runs"
}
```

The example does not define all mandatory checks; omitted checks still follow the same record shape. Deterministic ordering and stable IDs/codes are required. For bare `doctor`, the required checks are effective config parse/validation and readability of an **existing** SQLite catalog. Explicit missing/invalid config and unreadable/corrupt existing SQLite are `FAIL`; absent default config with valid effective defaults and absent first-run SQLite are `WARN`, never created or called healthy. Model execution and optional integrations cannot fail local catalog readiness: missing credentials are `WARN`; configured credentials mean only `configured, not probed`, not `OK: reachable`. A vector store that cannot be inspected without mutation is `SKIP` with a reason. `ok` means no required check failed; `WARN` or optional `SKIP` can coexist with `ok: true`. Summary status is `FAIL` if any required check fails, otherwise `WARN` if any warning or unverified optional `SKIP` exists, otherwise `OK`. A skipped required check due to inspection error must fail rather than silently mark healthy. A healthy live probe is a separate, explicit claim, not part of bare `doctor`. Subcommands follow the same envelope, with a `report` field only when applicable; preserve underlying issue codes.

Exit codes for new `doctor` paths: `0` for completed `OK` or `WARN`; `1` for completed diagnosis with a required `FAIL` (JSON report still on stdout); `2` for argparse usage errors; `3` for an unexpected inability to finish inspection (bounded error on stderr, no success JSON). A known missing/invalid config is a diagnosed failure (`1`), not an unexpected exception. No claimed success if the report was incomplete. `--json` errors use one sanitized JSON object on stderr and no stdout. Existing public aliases retain **their existing output and exit codes**; do not silently change them to the new envelope. Explicit opt-in `--live` failures produce a completed failure report rather than disguising it as an untested offline status.

## Compatibility and ownership

- Keep `--print-settings`, `catalog-check`, `diagnostics embeddings|analytics`, `prompt-validate`, `prompt-lint`, `prompt-test`, and `prompt-chain-validate` callable. Do not rewire aliases to a new exit code or JSON schema. First extract shared pure diagnostic logic, then add new parser/handler adapters. Legacy manager-backed aliases are not retroactively declared side-effect-free.
- Prefer `cli/doctor.py` for the new adapter/report formatting and a small pure health model (e.g. `core/doctor.py`). The settings loader remains the source of effective config; `core/catalog_check.py` owns catalog issue semantics. Do not put diagnostic decisions into argparse or build a second settings precedence model.
- `cli/parser.py` owns nested help and arguments; `main.py` dispatches `doctor` **before** manager initialization and missing-config creation prompts; `cli/commands.py` retains legacy handler entrypoints. Isolate the read-only SQLite adapter instead of adding a `readonly` flag to the normal repository constructor without proof.

### Legacy mapping (no silent behavior change)

| New route | Existing route retained | Boundary |
| --- | --- | --- |
| `doctor config --details` | `--print-settings` | New route uses sanitized doctor status/JSON; old route retains its text and exit semantics. |
| `doctor catalog` | `catalog-check` | Share pure issue detection only; legacy manager-backed startup and exit codes remain unchanged. |
| `doctor embeddings --live` | `diagnostics embeddings` | Explicit new backend-only probe does not initialize the manager/vector index; old route remains as-is. Default `doctor embeddings` never probes. |
| `doctor analytics --live` | `diagnostics analytics` | Current analytics snapshot invokes an embedding probe; new no-`--live` route must not. |
| `doctor prompt <ref> validate|lint|test` | `prompt-validate`, `prompt-lint`, `prompt-test` | Targeted on-demand checks; existing outputs and exits remain stable. |
| `doctor chain <file> validate` | `prompt-chain-validate` | Same pure validation meaning; existing output/exit preserved. |

The new route must not simply invoke the old CLI handler when doing so would build a writing manager or contact a provider. Do not promise JSON parity between old and new routes; explicitly version the doctor envelope and test both contracts.

## Bounded implementation and documentation order

1. **Doctor bootstrap + contract (delivered as `7487bbf`).** Real-process tests cover `doctor`, `doctor --json`, invalid explicit config, no default config, exit codes, and no writes/TTY prompts in a temporary HOME/config/SQLite/Chroma setup. `main.py` dispatches module execution before importing provider-backed core/GUI; installed `prompt-manager` uses `cli.entrypoint:main` to do likewise. Effective config is loaded through existing settings precedence; a shallow existing SQLite schema is checked through an immutable read-only URI. Existing non-empty WAL is reported as unverified rather than silently bypassed. The four checks distinguish configuration, local catalog, embeddings, and model execution; vector state is `SKIP`, not claimed healthy. Verified locally: 129 focused entry/help/doctor tests; full suite 913 passed, 1 skipped, core coverage 81.68%; Ruff lint/format, full strict Pyright, uv lock, wheel entrypoint inventory and diff check passed. No live provider probe. This is **shallow health only**, not the later `doctor catalog` record audit. Files: `cli/parser.py`, `main.py`, `cli/doctor.py`, `cli/entrypoint.py`, `pyproject.toml`, `tests/test_doctor_cli.py`, related help tests and docs.
2. **Read-only catalog (delivered as `4e8c28f`).** `doctor catalog [--json]` hydrates prompts/chains through `core/repository/read_only_catalog.py` without constructing a writing repository. It uses the pure checker for stable `CAT001–CAT006` findings; the new envelope includes counts, code/severity, and identifiers but omits legacy messages and prompt content. Missing first-run DB is `WARN`/0, incomplete or corrupt DB and any pending nonempty WAL are diagnosed `FAIL`/1; the latter must be retried after writers checkpoint/close, not silently read from a stale main file. An immutable SQLite read avoids sidecar writes. Isolated process tests cover first-run, corrupt/incomplete DB, rollback lock, nonempty WAL byte equality, chain reference issues, and JSON flag placement; the same fixture yields matching legacy issue codes/counts without changing the legacy route. Vector index is not inspected or called healthy. No provider run. Files: `core/repository/read_only_catalog.py`, `cli/doctor.py`, `cli/parser.py`, entrypoints, `tests/test_doctor_cli.py`, docs. Exact-SHA delivery verified in Quality Gates run `36004718272`.
3. **Config, embeddings and report detail (offline delivered as `4e8c28f`).** Offline `doctor config [--details] [--json]` reports effective settings validity and allowlisted source/availability booleans without raw path, model, DSN, credential or exception values. Offline `doctor embeddings [--json]` reports configuration readiness only, without vector/backend construction, network, or an OK connectivity claim. `doctor analytics [--json] [--export-csv PATH]` counts local execution successes/total in an immutable, sidecar-free SQLite snapshot; it is explicitly a report, not health. Missing catalog warns; corrupt/unreadable/pending-WAL catalogs fail. CSV writes only after an explicit path, exclusively creates the file, and refuses to overwrite; no record contents or model names are included. Early module/installed entrypoints, first-run, explicit invalid config, config-embedded ignored credentials, canonical dotenv credentials, JSON flag placement, WAL, and CSV error boundaries have real-process tests. The opt-in `doctor embeddings --live` is implemented after separate approval; `doctor analytics --live` is an additional locally verified slice, reusing the same bounded backend probe without opening the index; no real provider call was made for this slice. Do not rewire legacy aliases. Files: `cli/doctor.py`, `cli/parser.py`, `core/repository/read_only_analytics.py`, entrypoints, `tests/test_doctor_cli.py`, docs.
4. **Targeted validators and migration (delivered as `4e8c28f`).** `doctor prompt <uuid-or-exact-name> validate|lint [--json]` and `doctor prompt <uuid-or-exact-name> test --suite PATH [--json]` use the read-only loader and existing pure prompt validation/lint/test helpers; doctor fixture rendering uses a dedicated Jinja sandbox with built-in globals removed and accepts only bounded literal text plus scalar `{{ variable }}` substitutions. Loops, expressions, filters and attributes fail as fixture cases; the unchanged legacy runner supports its prior broader template semantics. The suite text is capped at 65,536 characters with at most 64 cases. `doctor chain <definition-file> validate [--json]` uses the existing model parser for structural validity and step count (not the legacy compatibility-field advisory report). New versioned reports include generic codes and counts, not raw messages, prompt names/bodies, definition contents, or suite values. JSON parse errors on doctor nested and root routes emit a bounded stderr object without echoing input arguments. Tests cover UUID and exact-name ambiguity, valid/invalid fixture suites, sandbox refusal of environment/filesystem access and expensive expressions, errors, JSON, family/leaf help, immutable stored DB, and issue-code parity with legacy real-process validators. Existing aliases retain their output and exits; no removal schedule.
5. **Public docs and closeout (delivered as `4e8c28f`).** `README.md` quick start promotes `prompt-manager doctor`; `docs/README-DEV.md` indexes available doctor routes while legacy commands remain supported; `docs/CHANGELOG.md` and `docs/STATUS.md` distinguish local verification from delivery. `docs/product-ssot.md` remains unchanged. Family/leaf help, real installed/module invocations, provider-free fixture behavior and legacy alias checks have been tested. Legacy aliases remain visible; removing them is a separate decision. Exact-SHA Quality Gates [36004718272](https://github.com/voytas75/PromptManager/actions/runs/36004718272) succeeded for `4e8c28fbb262ca00ec519c5ca5f8bef9de629a55`.

For each slice: focused tests → real isolated CLI (stdout/stderr/exit captured) → `.venv/bin/ruff check .` → `.venv/bin/ruff format --check .` → full configured `.venv/bin/pyright` → `.venv/bin/pytest -q -n auto --cov=core --cov-report=term --cov-fail-under=80` → `uv lock --check` and `git diff --check`. Refresh this ledger with Implemented/Verified and current next slice after each completed stage. No live provider acceptance without a separate explicit budget/approval for `--live`.

## Stop conditions and decision trigger

Stop if `doctor` construction creates files/collections, normalizes user data, prompts at a TTY, prints credentials, or invokes a provider without explicit `--live`. Stop if an alias's exit/output changes silently. Prefer the smaller alternative — keep old commands and add only a router/landing `doctor` — if read-only catalog loading requires invasive persistence changes or nested argparse adapters duplicate domain logic. Observable trigger: an isolated no-mutation test or alias parity test cannot pass without widening scope.

**Current next decision:** `doctor index` is locally verified for Chroma 1.5 SQLite metadata parity only (972 passed, 1 skipped; core coverage 81.66%; Ruff, strict Pyright, lock and wheel). Installed CLI on the current local catalog returned `INDEX_METADATA_MATCH`, 93 exact matching IDs, zero differences, status `WARN`, with `vector_index=not_verified`. No Chroma client or provider was invoked by the reader; HNSW vectors/search remain unverified. Independent read-only review found no blockers; local commit and remote delivery remain pending. The prior opt-in embedding and analytics `--live` backend probes were each accepted once and delivered (`355e650`, `40fd155` respectively), with exact-SHA CI in runs `36008958726` and `36016455242`; neither established index health. After this bounded index slice, legacy alias retirement needs a separate public-contract decision. Do not combine it with Dependabot or dependency changes.
