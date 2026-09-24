# PromptManager — `doctor` CLI contract and migration v1

Status: stage 1 locally verified; stages 2–5 pending
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
prompt-manager doctor embeddings [--json] [--live]
prompt-manager doctor prompt <uuid-or-exact-name> validate [--json]
prompt-manager doctor prompt <uuid-or-exact-name> lint [--json]
prompt-manager doctor prompt <uuid-or-exact-name> test --suite <file> [--json]
prompt-manager doctor chain <definition-file> validate [--json]
prompt-manager doctor analytics [report options] [--live] [--export-csv PATH]
```

- Bare `doctor` is a fast, bounded, **read-only and provider-free** assessment. It reports independent `config`, `local_catalog`, `search_embeddings`, and `model_execution` capabilities with `OK` / `WARN` / `FAIL` / `SKIP` (not assessed). It lists only actionable findings and a short recommended next step; verbose detail belongs to a subcommand. Offline LiteLLM means `model_execution: WARN` while a usable local catalog remains usable; never turn optional TTS, Redis, inference routing, or absent LLM credentials into a false catalog failure. Do not claim successful search or model access without a real probe: distinguish configured/readiness from reachability.
- Bare `doctor` inspects settings and existing paths directly, without `build_prompt_manager`, `PromptRepository(...)`, Chroma `PersistentClient`/`get_or_create_collection`, SQLite schema creation, network, seed data, interactive prompts, or automatic repair. If a config path is explicitly supplied but missing/invalid, report a bounded config `FAIL` without offering to create it. If the default config is absent and defaults suffice, report that fact without creating it. Inspect an existing SQLite DB using a read-only URI; check only bounded structural/readability signals. A missing DB on first-run is not silently created or called corrupt. If the vector store cannot be inspected without writes, mark it `SKIP`/unverified, not `OK`.
- `doctor config` uses the **same effective settings precedence** as application startup but has a separate health classification by capability. `--details` adds source-of-value/precedence and paths while sanitizing credential values, DSN userinfo/query, and exception messages; do not copy unredacted `--print-settings` output into JSON. Never emit secret prefixes/suffixes. No credential existence check is a network validation.
- `doctor catalog` reuses pure `core.catalog_check.run_catalog_check` but loads prompts/chains through an explicitly read-only repository path. Its issue codes and semantic findings stay unchanged; no manager creation. It may report vector-index inspection as `SKIP` until a proven read-only Chroma path exists. It should distinguish an empty/missing first-run catalog from a corrupt/unreadable existing one. Do not claim storage consistency from equal counts alone; any eventual index check must compare exact identifier sets.
- `doctor embeddings` without `--live` checks effective configuration and local existing artifacts only, not embedding generation. `--live` is an explicit opt-in to an actual backend probe; it may call a paid/remote provider and must say so in help before execution. Preserve the existing `diagnostics embeddings` legacy behavior separately. A future `--fix` is out of scope and requires its own risk decision.
- `doctor prompt ...` and `doctor chain ...` remain **on-demand checks**, never part of bare `doctor`. Reuse pure validators/test/lint functions and a read-only loader where persisted prompt data is needed; no provider calls. `doctor analytics` is explicitly marked `report (not health)`; no part of it influences bare `doctor`. `core.analytics_dashboard.build_analytics_snapshot` currently invokes `manager.diagnose_embeddings()` → `provider.embed(...)`, so default `doctor analytics` must bypass that path (or split out a safe analytics-only snapshot); `--live` opts into the embedding probe with cost warning. CSV export writes only when `--export-csv PATH` is explicitly requested, and help must state this exception.
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
| `doctor embeddings --live` | `diagnostics embeddings` | Old route's active probe remains as-is; new default `doctor embeddings` never probes. |
| `doctor analytics --live` | `diagnostics analytics` | Current analytics snapshot invokes an embedding probe; new no-`--live` route must not. |
| `doctor prompt <ref> validate|lint|test` | `prompt-validate`, `prompt-lint`, `prompt-test` | Targeted on-demand checks; existing outputs and exits remain stable. |
| `doctor chain <file> validate` | `prompt-chain-validate` | Same pure validation meaning; existing output/exit preserved. |

The new route must not simply invoke the old CLI handler when doing so would build a writing manager or contact a provider. Do not promise JSON parity between old and new routes; explicitly version the doctor envelope and test both contracts.

## Bounded implementation and documentation order

1. **Doctor bootstrap + contract (locally verified).** Real-process tests cover `doctor`, `doctor --json`, invalid explicit config, no default config, exit codes, and no writes/TTY prompts in a temporary HOME/config/SQLite/Chroma setup. `main.py` dispatches module execution before importing provider-backed core/GUI; installed `prompt-manager` uses `cli.entrypoint:main` to do likewise. Effective config is loaded through existing settings precedence; a shallow existing SQLite schema is checked through an immutable read-only URI. Existing non-empty WAL is reported as unverified rather than silently bypassed. The four checks distinguish configuration, local catalog, embeddings, and model execution; vector state is `SKIP`, not claimed healthy. Verified locally: 129 focused entry/help/doctor tests; full suite 913 passed, 1 skipped, core coverage 81.68%; Ruff lint/format, full strict Pyright, uv lock, wheel entrypoint inventory and diff check passed. No live provider probe. This is **shallow health only**, not the later `doctor catalog` record audit. Files: `cli/parser.py`, `main.py`, `cli/doctor.py`, `cli/entrypoint.py`, `pyproject.toml`, `tests/test_doctor_cli.py`, related help tests and docs.
2. **Read-only catalog.** Prove existing SQLite with WAL and missing/corrupt/locked states without creating sidecars or collections. Add read-only record loading and feed the existing catalog checker; compare its report to legacy `catalog-check` on the same isolated fixture. Prove `doctor catalog` causes no user-state changes and preserves stable catalog issue codes. Files: a small read-only adapter in `core/repository/` (exact location after inspection), `cli/doctor.py`, `tests/test_doctor_cli.py`, and existing catalog-check entry tests in `tests/test_main_entry.py` (or a new focused test file).
3. **Config, embeddings and report detail.** Add `doctor config [--details]`, `doctor embeddings` local-only, explicit `doctor embeddings --live`, and `doctor analytics` with a safe analytics-only path by default and an explicit `--live` embedding probe; test sanitization (including DSN), accidental calls, status boundaries, explicit export, and failure channels. Verify existing aliases unchanged with real subprocess tests. Files: `cli/doctor.py`, `cli/settings_summary.py` only if shared safe primitives are extracted, `cli/parser.py`, existing diagnostics/settings tests.
4. **Targeted validators and migration.** Add nested prompt/chain parsers and thin adapters to current pure checkers/read-only store; test UUID/exact-name ambiguity, suite input, JSON, parser help and historical alias parity. Add compatibility assertions for every legacy spelling; no deprecation/removal schedule yet.
5. **Public docs and closeout.** Once code passes, change `README.md` quick start to recommend `prompt-manager doctor`; update `docs/README-DEV.md` command index with doctor paths and label legacy names as supported; update `docs/CHANGELOG.md` and `docs/STATUS.md` with verified delivery, not forecast. Keep `docs/product-ssot.md` unchanged unless the product direction changes. Keep the near-term note a pointer, not a second feature contract. Do not hide old commands from help until nested help/index tests and real consumers are reviewed; removing aliases is a separate approval decision.

For each slice: focused tests → real isolated CLI (stdout/stderr/exit captured) → `.venv/bin/ruff check .` → `.venv/bin/ruff format --check .` → full configured `.venv/bin/pyright` → `.venv/bin/pytest -q -n auto --cov=core --cov-report=term --cov-fail-under=80` → `uv lock --check` and `git diff --check`. Refresh this ledger with Implemented/Verified and current next slice after each completed stage. No live provider acceptance without a separate explicit budget/approval for `--live`.

## Stop conditions and decision trigger

Stop if `doctor` construction creates files/collections, normalizes user data, prompts at a TTY, prints credentials, or invokes a provider without explicit `--live`. Stop if an alias's exit/output changes silently. Prefer the smaller alternative — keep old commands and add only a router/landing `doctor` — if read-only catalog loading requires invasive persistence changes or nested argparse adapters duplicate domain logic. Observable trigger: an isolated no-mutation test or alias parity test cannot pass without widening scope.

**Current recommended next execution slice:** stage 2 (safe read-only `doctor catalog` record-level audit). Stage 1 has been locally verified but is not a completed remote delivery until the exact commit and CI are confirmed; stages 3–5 remain pending. The listed nested routes are a contract proposal, **not yet callable commands**.
