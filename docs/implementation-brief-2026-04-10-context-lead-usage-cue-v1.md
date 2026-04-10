# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Context Lead Usage Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded **Context Lead Usage Cue** improvement inside the existing prompt detail flow so the operator can still see one compact `When to use` cue when a prompt has no credible saved scenario, description, or example cue, but the opening of the prompt body itself already contains a short usable usage signal.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse.

It should reduce friction between:
- "I opened the prompt detail view"
- and
- "I can quickly tell whether this prompt is probably the right one without reading the whole body."

## Scope

### In scope
- one bounded fallback path for the existing `When to use` cue in the shared prompt detail widget
- derive the fallback only from existing prompt body text (`context`)
- prefer the opening line or opening sentence of the prompt body when it is short and credible
- keep the current detail flow, labels, and rendering structure intact
- focused deterministic regression coverage for the new fallback and for weak-signal suppression

### Out of scope
- new metadata fields or schema changes
- generated advice or assistant-like recommendations
- new panels, chips, cards, or summary widgets
- broad redesign of prompt detail layout
- changes to retrieval ranking, prompt list behavior, or promote flow
- chains, analytics, sharing, voice, or web-enriched execution work

## Recommended UX posture

Prefer one quiet fallback inside the existing `When to use` surface.

Suggested v1:
- keep the current priority order for usage cues:
  1. scenario
  2. description
  3. example input
- add exactly one fallback after those:
  4. compact lead-in from prompt body
- use the fallback only when the prompt body begins with a short credible signal
- if the body opening is noisy, generic, or too long, show nothing and keep the current quiet behavior

Default recommendation:
- **one bounded fallback heuristic**, not a richer summary system

Reason:
- low scope
- stays inside the existing detail seam
- improves inspect clarity without inventing new data or behavior

## Data source

Use existing prompt data only.

Preferred source for the new fallback:
- `prompt.context`

Preferred extraction posture:
- inspect only the opening line or opening sentence
- flatten whitespace
- trim common markdown markers or prompt-body prefixes when safely possible
- reject long or weak-signal results

Do not add new persistence in this slice.

## Likely implementation seam

### UI / behavior
- `gui/widgets/prompt_detail_widget.py`
  - extend `_resolve_usage_cue()` with one bounded `context` fallback
  - keep the existing render path unchanged

### Tests
- `tests/test_prompt_detail_widget.py`
  - add one positive test for a compact context-derived cue
  - add one negative test for noisy or weak context openings if needed

## Happy-path scenario

1. User opens a prompt in the existing detail view.
2. The prompt has no saved scenario, no credible short description cue, and no usable example cue.
3. The prompt body starts with a short credible line or sentence that already explains when the prompt is useful.
4. The detail view shows that text in the existing `When to use` cue.
5. The operator can judge fit faster without reading the whole prompt body preview.

That is enough for v1.

## Acceptance checks

1. The existing detail view can show a `When to use` cue derived from prompt body lead-in when higher-priority saved cues are absent.
2. The fallback uses only existing prompt body text.
3. Weak, noisy, or overly long prompt body openings do not create a fabricated or cluttered cue.
4. No new schema or new UI surface is introduced.
5. Focused regression coverage protects:
   - context-derived cue visible when the lead-in is credible
   - cue remains hidden when the context opening is weak
   - existing detail labels and flow remain bounded

## Suggested test

One focused test addition is enough.

Recommended shape:
- create one prompt with empty scenarios, empty useful description, no example input, and a short body lead-in like `Use when summarizing deployment risks for the release handoff.`
- verify the existing detail view exposes that text in `When to use`
- add one weak-signal case such as `Prompt:` or a long noisy opening and verify no new cue is shown

## Rollback

Rollback should be one isolated patch:
- remove the new context fallback from `_resolve_usage_cue()`
- remove the focused regression test additions
- leave the rest of the detail view, usage cue order, and inspect flow untouched

## Anti-goals

- do not create a general prompt-summary engine
- do not add AI-generated usage hints
- do not redesign the detail panel
- do not change the meaning of scenarios or description fields
- do not widen this into retrieval, reuse, or ingest work outside the existing detail seam

## Notes for implementation

- Keep the slice boring.
- Prefer deterministic text cleanup over smart inference.
- Improve inspect confidence, not feature breadth.
- If the heuristic starts needing too many exceptions, stop and keep the current behavior.
