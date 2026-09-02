# PromptManager — Implementation Brief

Date: 2026-04-12
Status: ready
Feature: Template-Aware Workspace CTA v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-11-template-variable-cue-v1.md`
- `docs/implementation-brief-2026-04-11-template-workspace-handoff-cue-v1.md`

## Goal

Implement one bounded **Template-Aware Workspace CTA** improvement in the existing prompt detail flow so a prompt with detected template variables makes the next reuse action more obvious **at the point of decision**, without adding a second primary button.

The desired experience is simple:

> this prompt has variables -> I can immediately tell that `Open in Workspace` is the right next step

This slice should strengthen the existing handoff, not create a parallel template workflow.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse.

PromptManager already has:
- a visible `Requires variables: ...` cue,
- a template-aware `Open in Workspace` tooltip.

This slice should answer the next practical question more clearly:

> when I see a template prompt, which action should I use right now?

The answer should remain:

> use the existing `Open in Workspace` handoff

That should become more legible without adding a second CTA, a template editor entry point, or a new flow.

## Scope

### In scope
- strengthen the existing `Open in Workspace` affordance for prompts with detected template variables
- keep the same primary action path and destination mental model
- improve legibility at the click decision point, for example through one bounded visible cue near the existing action area
- reuse the same variable-detection seam already used by `Template Variable Cue v1` and `Template Workspace Handoff Cue v1`
- keep changes local to the shared detail-view reuse seam
- add focused regression coverage for template-aware visible CTA behavior and unchanged plain-prompt behavior

### Out of scope
- adding a second primary button such as `Use Template` or `Work with Template`
- changing `Open in Workspace` destination semantics
- changing `Copy Prompt` semantics
- adding variable editing in detail view
- adding a template panel, wizard, or inline form
- changing workspace execution behavior
- changing template preview behavior
- broad reuse-surface redesign
- schema or persistence changes

## Recommended UX posture

Prefer **one smarter handoff** over **two competing actions**.

Suggested v1 posture:
- keep `Open in Workspace` as the single reuse handoff for template prompts
- for prompts with variables, add one subtle but visible clarification near that action so the operator does not need to infer the next step from tooltip only
- keep the cue short and action-oriented
- preserve the current plain-prompt path unchanged

Possible acceptable v1 shapes:
- a small helper line near the reuse actions such as:
  - `Has variables. Open in Workspace to fill and work from a copy.`
- a bounded template-aware secondary label/caption near the existing button
- a modest state emphasis that makes the workspace handoff feel intentional for templates

Default recommendation:
- **make the right action more legible, not broader**

## Decision rule

If a visible enhancement starts to look like a second action, the slice is drifting.

This slice is successful only if the operator can more confidently choose the existing workspace handoff **without** being asked to choose between two overlapping controls.

## Likely implementation seam

### Detail view reuse area
- `gui/widgets/prompt_detail_widget.py`
  - extend the existing template-aware reuse affordance
  - keep `Open in Workspace` as the primary control
  - add one bounded visible clarifier only when detected variables exist
  - preserve plain-prompt behavior exactly where possible

### Tests
- `tests/test_prompt_detail_widget.py`
  - cover visible template-aware CTA clarification for template prompts
  - cover absence of the extra cue for plain prompts
  - cover unchanged action labels and enabled-state behavior

## Happy-path scenarios

### Scenario A: template prompt in detail view
1. User opens a prompt whose body contains detected template variables.
2. The detail view already shows the variable cue.
3. Near the reuse actions, the UI makes it more explicit that `Open in Workspace` is the next action for filling and using the prompt.
4. The user can choose the correct handoff without guessing.

### Scenario B: ordinary prompt in detail view
1. User opens a prompt with no detected variables.
2. The reuse area behaves as it does today.
3. No extra template-oriented cue appears.

### Scenario C: template prompt still keeps one action path
1. User sees a template prompt.
2. The UI clarifies the existing workspace handoff.
3. The UI does not add a second competing template button.

That is enough for v1.

## Acceptance checks

1. A prompt with detected template variables shows one visible, bounded clarification that `Open in Workspace` is the correct next handoff.
2. The main action remains `Open in Workspace`; no second competing primary CTA is introduced.
3. Plain prompts do not show the template-specific visible cue.
4. Existing labels `Copy Prompt` and `Open in Workspace` remain unchanged unless a tiny local wording adjustment is strictly required.
5. Existing action semantics remain unchanged.
6. Focused regression coverage protects template-aware visible CTA behavior and ordinary-prompt fallback behavior.
7. No panel, wizard, inline variable editor, or broader workflow is introduced.

## Rollback

Rollback should be one isolated patch:
- remove the visible template-aware CTA clarification,
- remove any tiny helper or conditional rendering added only for this slice,
- remove the focused regression tests,
- leave the existing variable cue, tooltip behavior, and reuse actions untouched.

## Anti-goals

- do not add a second primary button
- do not create a `Use Template` workflow
- do not redesign the reuse area broadly
- do not add variable editing to detail view
- do not auto-open template preview or workspace flows
- do not touch search, capture, fork, or template-preview seams in this slice

## Notes for implementation

- Keep the slice boring.
- The operator should feel more certain about the current handoff, not asked to learn a new one.
- Prefer one visible clarifier over richer explanation.
- If implementation starts introducing multiple template-specific actions, the slice has gone too far.
