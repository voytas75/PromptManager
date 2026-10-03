# PromptManager — one asset-loop task probe

Status: discovery and approved wording follow-up completed locally; publication authorized. Exact-SHA remote delivery/CI receipts are reported separately after push.
Baseline: `master@b943b0392f38c4f68098e7f2a77595ade4d1d3b5`, initially clean worktree.
Authority: `docs/product-ssot.md` and the active near-term planning note.

## Approved scope

One synthetic incident-handoff task: find an existing prompt, inspect its body/input requirements and evidence, then select non-executing reuse or inspect edit/fork. Approval covers diagnosis and this factual checkpoint, not implementation or publication.

- Product source, tests, settings, dependencies, CI and user catalog unchanged.
- One new checkpoint, at most 100 lines; executable probes and data in managed scratch.
- No paid models, real provider/search traffic, user clipboard mutation, native Windows edits, staging, commit or push.
- Scratch HOME/CWD/config/cache, sanitized provider environment, fail-on-network/write-outside-scratch guards, local SQLite and injected index/embedding doubles.

## Acceptance and evidence limits

- Real manager/search coordinator hydrates injected index hits from the local catalog.
- Real list model, detail widget and history controller display requirements/action cues.
- Real workspace/copy actions preserve stored body; copy uses an injected clipboard sink.
- Synthetic catalog record set, file set and file bytes unchanged during read/handoff.
- Edit/fork menu inspected without executing either mutation.

Injected hits prove hydration/handoff, not semantic ranking quality. Offscreen widgets prove behavior/text, not native visual usability or real-user completion time. Manual local rendering proves the template can fill, not that the complete workspace rendering UI was exercised. Synthetic fixtures are not user-demand evidence.

## Executed scenario and result

Task: locate `Incident handoff` for an incoming on-call summary; inspect requirements; open the unchanged template in Workspace, without Run. Fixture requires `audience` and `incident` and has zero usage/ratings and no run history.

- Composed probe: exit 0, **16/16 checks PASS**.
- One local embedding invocation and one injected index query; **0 network attempts, 0 prompt executions**.
- Requirements shown: `Requires variables: audience, incident`.
- Detail evidence: `No run evidence yet`; next action: `Validate before reuse`.
- Workspace tooltip names both inputs; handoff transfers exactly the stored body without executing.
- Copy returns the stored template, not separately rendered output. This is not a defect.
- Fork tooltip: `Create a fork linked to this prompt and open it for editing.`
- Duplicate tooltip: `Create an editable copy of this prompt without fork lineage.`
- Missing Edit tooltip alone is not a reproduced task failure.

## Reproduced gap: retrieval readiness wording

The same fixture simultaneously displays:

- list handoff: **`Ready to reuse`**;
- list fit: **`No run evidence yet`**;
- detail decision: `Reuse as-is` (no body edit recommended);
- detail next action: **`Validate before reuse`** plus required inputs.

Independent replication through `PromptListDelegate.handoff_cue_text()` and `fit_cue_text()` confirmed both list strings with usage_count=0 and rating_count=0.
Root seam: `gui/prompt_list_model.py:183-193` returns `Ready to reuse` for title match + non-draft, without assessing run evidence or template inputs. `gui/prompt_list_delegate.py:48-55,71-75` consumes this role as visible row text.
This is a demonstrated wording inconsistency, not a failed search/handoff, proven wrong model output, or observed user harm. Detail guidance mitigates it; neighboring tests currently protect this behavior and remain green.

## Verification

Provider-blocked project-interpreter run of these existing suites: `test_prompt_list_model.py`, `test_prompt_list_coordinator.py`, `test_prompt_actions_controller.py`, `test_prompt_detail_widget.py`, `test_workspace_history_controller.py`, `test_template_preview_widget.py`.
Result: **128 passed in 0.52s**, exit 0, 0 network attempts; count independently checked against JUnit testcases. No full-suite, native Windows, live provider or new CI claim.
Harness startup corrections: supplied required fixture ID and redirected pytest's configured log_file into scratch after the write guard blocked its default target. Neither was a product failure; final verification exited 0.
Scratch evidence: `/home/voytas/.hermes/cache/scratch/promptmanager-asset-loop-liu7wpwi/` (`probe.py`, `receipt.json`, `process.json`, `verify.py`, `guidance-replication.json`, `test-status.json`, `junit.xml`). Scratch expires; the factual receipt summary above is durable in this checkpoint.

## Decision / proposed next slice

Recommend a separately approved GUI-local wording repair: replace readiness-as-validation language on the title-match handoff with neutral inspect-first guidance. Prefer reuse of existing `Inspect before reuse` copy over inventing validation heuristics, ranking, persistence, new fields or automatic runs. Acceptance: one regression covers the reproduced no-history template, then adjacent list/detail/action tests stay green and non-executing handoff is unchanged.
Strongest alternative: leave wording as-is because `Ready` means asset availability, not validated output. Change the recommendation if native/user acceptance establishes that this distinction is consistently understood; currently that is unverified.
At discovery closeout, implementation, commit, push and native/provider acceptance required separate approval. No automatic new campaign.

## Approved follow-up — title-match inspect-first wording

User approved the described wording slice. Local implementation and independent review completed. Commit/push of the four-file slice is authorized; remote delivery/CI is a separate post-push verification, not inferred from local tests. Only `gui/prompt_list_model.py`, its existing test module, this checkpoint and one changelog entry are in scope. Preserve the prior discovery receipt above; do not implement readiness heuristics or change search, ranking, drafts, detail decisions, workspace/copy actions, CLI, settings or persistence. Provider and native Windows acceptance remain unapproved.
Acceptance: template-without-history regression fails on old copy, passes on `Inspect before reuse`; existing plain/title, draft and non-title paths remain covered; adjacent tests, Ruff, strict file-scoped Pyright and the same synthetic composed handoff smoke pass. Baseline Pyright for both selected Python files: 0 errors, 0 warnings. Evidence directory: `/home/voytas/.hermes/cache/scratch/promptmanager-inspect-first-dljuu4mh/`.

- RED: the new template regression plus updated plain-title test both failed exactly on `Ready to reuse` versus `Inspect before reuse` (2 assertion failures, no network attempts).
- GREEN: entire `tests/test_prompt_list_model.py` — 27 passed in 1.85s; only production change is one returned string, plus the module history entry.
- Adjacent: the six discovery suites plus `test_prompt_list_presenter.py` and `test_retrieval_cues_parity.py` — 137 passed in 2.18s, 0 network attempts; counts independently verified from JUnit.
- Re-smoke: retained synthetic composed search/detail/workspace/copy probe plus two list-cue checks — 18/18 PASS, exit 0, no stderr, 0 network attempts, 0 executions; stored template and catalog bytes/record set unchanged. Row now says `Inspect before reuse`; missing evidence, input requirements and validation-first detail guidance remain intact.
- Static: `.venv/bin/ruff check --no-cache .`, focused `ruff format --check`, strict `pyright gui/prompt_list_model.py tests/test_prompt_list_model.py`, syntax parse and `git diff --check` all passed; Pyright 0 errors/warnings, same as baseline. Added-line scan found no new secrets or risky execution/I/O surface.
- Coverage: network-blocked list suite with `--cov=/absolute/repo/gui`; selected model 128/147 statements covered (87.07%), changed executable line 191 covered. Directory-wide coverage is not claimed or gated; the selected-file >=80% criterion is independently asserted from JSON.
- Harness limitation: dotted `--cov=gui.prompt_list_model` crashed during cold native import/discovery; preloading it produced misleading missing-import coverage. Filesystem-directory source collected normally (27 passed in 3.67s) without changing dependencies, tests, CI or coverage settings. This is not a product/native acceptance result.
- Updated Unreleased changelog only for the delivered GUI wording; product SSOT and CLI remain unchanged.
- Independent read-only review: PASS, no security/logic blockers. Review confirmed AST changes only the returned literal, aside from module history, with existing conditions/draft suppression preserved. Its checkpoint-closeout suggestion is addressed here; the parent separately resolved coverage using directory-source collection above. No publication performed.
- Local outcome closed; publication authorized after review. Remote SHA/CI readback will be reported in the delivery receipt. No next product slice is automatically approved; native acceptance remains a separate decision.
