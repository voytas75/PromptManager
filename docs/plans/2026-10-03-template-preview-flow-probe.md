# PromptManager — template preview operator-flow probe

Status: diagnosis completed; one reset defect reproduced, repair not approved.
Baseline: `master@8e9a6bdd04723cd731bf4af7f957e5733efd0f9d`; clean worktree at intake.
Authority: `docs/product-ssot.md`; priority 1 retrieval/inspect/reuse confidence.

## Approved scope

One real-Qt synthetic flow: select/inspect a variable template, open its workspace, fill missing inputs partially then completely, switch to a second prompt, return and clear selection. Inspect preview and Run availability, but never Run. No production or test-code changes, dependencies, CI, user catalog/settings, native Windows acceptance, commit or push. Only this factual checkpoint added; scripts/data stay in managed scratch.

## Evidence contract and limitations

- Real list selection signal, history/detail, template controller/widget and non-executing workspace action; Qt event loop in a fresh child process.
- Real variable editors changed with `insertPlainText`/`clear`; generated widget/event evidence, not physical keyboard/mouse acceptance.
- Scratch SQLite/QSettings, sanitized provider environment, fail-on-network/execute boundaries, unchanged catalog bytes/record sets.
- Injected index/embedding boundary: no semantic search or actual embedding invocation. Synthetic control availability enables Run buttons without constructing an executor; not provider readiness.
- Not full application launcher, native visual, actual Run/model output, schema-validation matrix or fresh-process QSettings restore acceptance.

## Executed sequence

A: `A incident: {{ incident }} / Audience: {{ audience }}`.
B: `B incident: {{ incident }} / Owner: {{ owner }}` (shares `incident` deliberately).

1. Select A, invoke real Open in Workspace button: stored template unchanged, required fields present, Run and mirrored shortcut disabled.
2. Fill `incident` only: still not ready, missing `audience` indicated.
3. Fill `audience`: exact rendered A text; `Preview ready.`, both Run controls enabled under synthetic availability.
4. Select B: fields and payload belong to B, no A values/text; both Run controls disabled until B is complete.
5. Fill B: exact rendered B text, both controls enabled.
6. Return A: own prior values restored, no B text/values; ready state matches A.
7. Remove required A input: ready state revoked, both controls disabled.
8. Clear selection through the real selection model: empty preview and disabled controls, but old A input fields and payload remain.

Refined composed receipt: **28/30 checks pass**, child exit 1 intentionally classifies the finding. Failed criteria: empty variable payload and no visible variable inputs after selection clear. **0 network, 0 execution, 0 embedding attempts**; catalog record set/file set/bytes unchanged.

## Confirmed defect: empty-selection reset leaves old inputs

After clear, `selected=None`, preview is empty, status says `Select a prompt to enable template previews.`, and both Run controls are disabled. However, the previous `audience`/`incident` editors remain visible and `variables_payload()` still returns `{'incident': 'A synthetic rollback'}`.
Independent minimal real-widget/controller reproduction confirms both empty-before-clear and filled-before-clear variants. Filled `recipient` remains visible/returned after `update_preview(None)`; returning to that prompt restores its own saved value normally. Reproduction exit 0 means the observed defect assertions passed, not that the product reset is correct. No run signal or network attempt occurred.
Root seam: `TemplatePreviewWidget.clear_template()` calls `set_template('', None)`. The empty branch in `gui/template_preview.py:96-100` resets `_variable_names` and preview but returns before `_rebuild_variable_inputs()` clears editor/label maps and removes old widgets. Retention survives posted DeferredDelete handling and normal event processing; this is not a transient Qt frame.
Impact proven: stale visible input state and stale public payload with no selected asset. No accidental model execution, catalog corruption or cross-prompt bleed demonstrated.

## Verification and corrected assumptions

- Existing nearby suites: `test_template_preview_widget.py`, `test_workspace_history_controller.py`, `test_prompt_actions_controller.py`, `test_prompt_detail_widget.py`, `test_main_window_bridges.py`: **94 passed in 3.98s**, exit 0, 0 network attempts; total verified from JUnit.
- Initial harness omitted required layout-state dependency; corrected with a local stub before classifying runtime behavior.
- Initial assertion expected all missing names in status. Real StrictUndefined rendering reports the first undefined variable and keeps fields marked missing. Reclassified as narrower truthful guidance, not the selected repair.
- Old field visibility on switch was initially sampled before deferred deletion/layout events; settled sampling proves A/B switch correct. Only empty-selection retention remains reproducible.
- Network-blocked composed probe plus minimal reset probe exercised real widgets, no source/test edits. Tracked Git diff remains empty; this checkpoint is the only untracked file. No new CI/native/provider claim.
Evidence: `/home/voytas/.hermes/cache/scratch/promptmanager-template-flow-wamxnrvs/` (`initial-receipt.json`, `refined-receipt.json`, `reset-reproduction.json`, process logs, `adjacent-junit.xml`). Scratch expires; summary here retains the result.

## Decision / next bounded repair (not approved)

Recommend empty-template reset on the existing widget seam: remove old variable widgets/label maps and make public payload empty when no template is selected; preserve per-prompt QSettings so returning to A restores its inputs. Cover clear-after-empty, clear-after-filled, disabled Run/shortcut, empty rendered view and restore-on-reselect with real-Qt regressions; keep search, execution, schema, persistence format and CLI unchanged. No additional missing-status wording change in this slice.
Strongest alternative: keep old fields as deliberate workspace draft state. Prefer it only if an explicit product contract names that state and separates it from the selected-prompt payload; currently no such contract is established, and the reset API/empty-selection message point to a clear state.
Diagnosis closed. Checkpoint-only commit/push is now authorized; exact-SHA remote delivery and CI are reported separately after publication. Reset implementation and native/model acceptance remain unapproved.

Daily approved scope: asset-loop diagnosis completed; title-match wording repair completed, independently reviewed and delivered at baseline SHA with successful Quality Gates; template-preview diagnosis completed with the reset defect recorded. No approved implementation task remains open. The recommended reset repair is a newly identified, separately scoped candidate, not a completed fix.
