# PromptManager — Intentional Capture Continuity v1

Status: completed (2026-09-22; historical execution ledger, not an active next slice)
Owner: Wojtek / Prompt Manager Team
Updated: 2026-09-22
Canonical product SSOT: `docs/product-ssot.md`
Canonical near-term plan: `docs/plans/2026-05-10-product-direction-ssot-next-cycle.md`

## Product decision

PromptManager should make preserving a useful prompt from outside the catalog as deliberate and low-friction as reusing one already inside it. The product remains local-first and asset-first: an explicit user action creates an editable draft preview, then the existing Quick Capture → Draft path owns review and persistence.

Every proposed slice must answer:

> Does this make an operator use their prompts more often and with more confidence?

If not, it does not enter the cycle.

## Scope

### In scope

- Add one explicit **Paste from Clipboard** action to the existing Quick Capture dialog.
- Read clipboard text only when the operator clicks that action.
- Put the text into the existing editable draft body; do not save automatically.
- Set source to `clipboard` only when the source field is blank, preserving manually chosen provenance.
- Explain that the content remains a reviewable draft preview.
- Retain typed draft content if the clipboard is empty and show a compact explanation.
- Reuse existing Quick Capture title derivation, normalization, draft metadata, validation, and promotion behavior.

### Out of scope

- Background clipboard monitoring, global OS hotkeys, startup clipboard reads, or automatic saves.
- New persistence/schema, provider calls, model-generated titles, external integrations, or a new capture flow.
- CLI capture commands, collection features, or changes to promotion semantics.

## Acceptance contract

1. Clicking the explicit action reads the current clipboard and replaces the editable body preview only when it has non-whitespace text.
2. The operator can review or edit the preview before the existing Save Draft action is available.
3. Blank source becomes `clipboard`; an existing manual source remains unchanged.
4. Empty clipboard preserves typed content and explains the condition.
5. `QuickCaptureDraft.build()` and the existing promotion path continue to own the resulting prompt contract.
6. Verification is deterministic and provider-free: focused GUI tests, then repository gates before any delivery decision.

## Completion record

Completed 2026-09-22:

- Added the explicit **Paste from Clipboard** action to the existing Quick Capture dialog.
- The action reads clipboard text only after the operator clicks, puts it in the editable draft preview, and marks blank provenance as `clipboard` without overwriting a manual source.
- Empty clipboard content preserves the existing preview and explains what the operator should copy.
- Verified with the focused provider-free Quick Capture suite (`32 passed`), Ruff lint/format, focused Pyright (`0 errors`), and `git diff --check`.

Deferred by design: background monitoring, global hotkeys, automatic save, provider enrichment, persistence/schema changes, and a parallel capture surface.
