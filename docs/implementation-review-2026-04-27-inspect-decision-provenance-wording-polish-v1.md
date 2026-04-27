# PromptManager — Implementation Review

Date: 2026-04-27
Target: `Inspect Decision-Provenance Wording Polish v1`
Expected source: `docs/implementation-brief-2026-04-27-inspect-decision-provenance-wording-polish-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. It keeps the current decision-provenance seam intact, shortens only the visible wording on that cue, and avoids widening into new provenance states, evidence logic, or inspect/detail redesign.

## What matches

### 1. The slice stays in the intended seam
The implementation remains local to the existing provenance wording path:
- `gui/workspace_history_controller.py`
- `tests/test_workspace_history_controller.py`
- `tests/test_prompt_detail_widget.py`
- `docs/CHANGELOG.md`

That matches the brief's bounded posture.

### 2. The same provenance categories remain intact
The delivered behavior preserves the same three provenance states:
- comparable-run evidence
- limited run evidence
- fork-lineage-only evidence

Only the visible phrasing changed. No branching logic or source selection changed.

### 3. The cue now reads shorter and more subordinate to `Decision`
The wording now follows a tighter pattern:
- `Based on latest 2 comparable runs`
- `Based on limited run evidence`
- `Based on fork lineage only`

That fits the intended role of provenance as a quiet source cue rather than a second recommendation.

### 4. Shared and template detail surfaces remain aligned
The focused controller coverage still locks both detail surfaces to the same provenance wording for:
- comparable-run states
- limited-evidence states
- lineage-only states

That preserves the parity posture of the inspect/detail flow.

### 5. Focused regression coverage matches the slice
Focused tests now cover:
- shared-widget visible provenance rendering with the shortened wording
- comparable-run provenance wording in controller paths
- limited-evidence provenance wording in controller paths
- lineage-only provenance wording in controller paths

Validation passed:
- `pytest -q tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py`
- result: `56 passed`

## What is missing

Nothing material relative to the brief.

The implementation did not need broader docs changes, widget changes, or new helpers beyond the local wording update because the slice was intentionally narrow.

## What drifted / widened

No meaningful scope drift is visible.

The implementation avoided:
- new provenance states
- new inspect/detail cues
- changed decision logic
- changed next-action logic
- changed run-summary logic
- broader inspect/detail wording cleanup outside this cue

That is the right outcome.

## What is unverified

### 1. Live visual feel in dense inspect/detail layouts
This review confirms wording alignment and regression coverage, but it does not include a manual GUI pass across cramped layouts or theme variations.

### 2. Whether the shortened wording is the final best phrasing
The new phrasing is clearly more compact and consistent, but this review does not measure operator preference between this exact wording and nearby alternatives.

That is acceptable for this slice because the goal was bounded wording polish, not a broader UX study.

## Recommended next action

Treat `Inspect Decision-Provenance Wording Polish v1` as delivered.

Do not widen it into new provenance surfaces unless later operator evidence shows the current single-line cue is still too opaque.

If a follow-up is ever needed, it should be another tiny inspect-clarity slice on the existing seams only.

## Sources reviewed

- `docs/implementation-brief-2026-04-27-inspect-decision-provenance-wording-polish-v1.md`
- `gui/workspace_history_controller.py`
- `tests/test_workspace_history_controller.py`
- `tests/test_prompt_detail_widget.py`
- `docs/CHANGELOG.md`
- focused validation result:
  - `pytest -q tests/test_workspace_history_controller.py tests/test_prompt_detail_widget.py` → `56 passed`
