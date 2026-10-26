# SQLite–Chroma–worker fault/recovery qualification

Status: complete (bounded diagnostic slice; no product repair)
Source revision: e861b0166a6e038cc89d609abd0349d6a74e26fe

## Approved scope

Provider-free disposable failure/recovery probes and checkpoint only. No product/test source edits, live catalog or Windows writes, providers, dependencies, CI changes, commit or push.

## Observed results

Four executed cases:
1. Create with fake index mutation then exception: SQLite row and first snapshot compensated away; cache empty; orphan index entry remains. Ordinary rebuild (0, 0) does not remove it. Reset rebuild (0, 0) removes it in the fake backend.
2. Public update with fake mutation then exception: canonical body/vector/version and cache retain prior state; index temporarily contains rejected document/vector. Ordinary rebuild (1, 0) restores consistency from SQLite.
3. Controlled worker interruption before persistence with another queued task: in-memory queue not restored on reopen. Existing rebuild (1, 0) restores embedding/index/cache; canonical body/history preserved.
4. Real disposable persistent Chroma with an intentionally missing index record: rebuild (1, 0) restores document and vector consistent with SQLite/cache (float32 comparison tolerance 1e-6).

These are partial-state observations, not evidence of loss of existing canonical catalog data. Failed creation is not retained. Fake mutation failures do not establish actual Chroma crash behavior; controlled worker stopping is not SIGKILL/process-crash qualification.

## Verification

Executed with project .venv Python, scratch working directory, dotenv disabled, isolated HOME/XDG and blocked network. Targeted suites: test_background_embedding_freshness.py, test_prompt_manager_reset.py, test_prompt_manager_storage.py. JUnit: 37 passed, 0 failed/errors/skipped. Guard: zero network attempts.

Parent verified receipts case count/names, JUnit totals, guard and unchanged clean source HEAD before checkpoint writes. No full suite or native GUI acceptance claimed.

Scratch evidence (ephemeral, not a durable repository test): /home/voytas/.hermes/cache/scratch/pm_fault_qualification/{probe.py,receipts.json,probe.log,tests.log,tests.xml,test-guard.json}.
Commands from scratch directory:
- /home/voytas/projects/PromptManager/.venv/bin/python -B probe.py
- /home/voytas/projects/PromptManager/.venv/bin/python -B probe.py tests

## Decision

Existing rebuild recovers missing/stale records, but ordinary rebuild is not complete orphan reconciliation. No broad refactor authorized or justified by this slice alone.

## Approved successor — real Chroma orphan/reset qualification

Status: complete. Real disposable Persistent Chroma and SQLite; deterministic embeddings; source HEAD unchanged. Two canonical prompts and three history snapshots were created through public manager operations. One orphan was inserted directly into the real index (controlled fixture, not actual crash).

Ordinary rebuild returned (2, 0) and retained all three index IDs. Reset rebuild returned (2, 0), removed the orphan and retained exactly the two canonical IDs. Full SQLite rows and three history snapshots remained identical; documents/vectors/cache were checked (vector tolerance 1e-6). A separate process reopened persistence and reproduced the exact final SQLite/index snapshot.

A wrapper raising before delete_collection caused public reset to report PromptStorageError without changing the snapshot or generating embeddings. This does not qualify failures after collection deletion or actual process crashes.

Execution evidence: 25 main assertions and 8 reopen assertions; main/reopen exit 0; zero attempted network connections. Parent independently verified script SHA-256, exact ID equality, unchanged full rows/history, final reopened snapshot, pre-delete refusal and guards. Prior pytest suites were not repeated.

Scratch evidence: /home/voytas/.hermes/cache/scratch/pm_fault_qualification/reset_real_v2/{probe.py,receipts.json,reopen.json,expected.json,probe.log}. Script SHA-256: 4c8e1d229d18bb98d52aa7cc8c621da387825b3a734fb99ff067a69f765510a5. First harness used a nonexistent history column; corrected only in scratch and rerun on a fresh disposable database.

Decision: existing reset rebuild qualifies orphan removal in this bounded real-backend scenario; no product repair or broad refactor follows automatically. Diagnostic queue closed. Actual crash, post-delete reset failure and concurrent-writer qualification remain outside this approval. Reset is not approved for a live catalog and may regenerate embeddings/contact configured providers.
