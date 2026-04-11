# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: README / Positioning Cleanup v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

Validation:
- docs-only pass
- spot-check: `README.md`, `docs/CHANGELOG.md`

## Goal

Implement one bounded **README / Positioning Cleanup** pass so the public repo story matches the current product center more closely: PromptManager as a **local-first canonical home for prompt assets**.

## Product intent

This slice strengthens product discipline rather than adding product surface.

It should reduce drift between:
- the product boundary SSOT,
- the ordered backlog,
- the restart brief,
- and the public-facing README.

The main aim is to keep future roadmap and implementation choices anchored to the prompt asset loop rather than to surrounding workstation-like surfaces.

## Scope

### In scope
- tighten README wording where secondary surfaces visually compete with the prompt-asset center
- make the README clearer about which capabilities are:
  - core,
  - supporting,
  - secondary / not product center
- keep the top-level product description aligned with the existing SSOT
- update changelog trail for the docs-only positioning pass

### Out of scope
- broad README rewrite
- changing product scope in SSOT files again
- code changes
- pruning features from the product
- UI redesign
- analytics, chain, sharing, voice, or execution implementation changes

## Recommended posture

Prefer one calm classification pass over a rhetorical rewrite.

Suggested v1:
- keep the strong top-of-README framing already in place
- reduce the below-the-fold impression that execution, analytics, chains, sharing, and voice are co-equal product centers
- describe these as useful supporting or secondary surfaces without pretending they do not exist

Default recommendation:
- **make the README more honest about product center, not more markety**

## Likely implementation seam

### README
- keep the opening product framing intact
- tighten the `Detailed features` section into clearer product-center buckets
- ensure advanced/secondary surfaces are still documented, but visually demoted

### Changelog
- add one short note recording the bounded README positioning cleanup

## Happy-path scenario

1. A new reader opens the repo.
2. They understand quickly that PromptManager is primarily a prompt catalog and reuse tool.
3. They still see execution, analytics, chains, sharing, and voice support, but those no longer read like the product center.
4. Future contributors inherit a cleaner public story.

That is enough for v1.

## Acceptance checks

1. README still truthfully documents major surfaces.
2. Core prompt-asset workflow reads as the dominant product story.
3. Supporting execution/history surfaces are described as supporting rather than central.
4. Secondary breadth (chains, broader analytics, sharing, voice, workstation-like helpers) is visibly demoted in wording or structure.
5. No broad docs rewrite or scope expansion is introduced.

## Rollback

Rollback should be one isolated docs patch:
- revert the README positioning/classification edits
- revert the changelog note
- leave backlog, boundary SSOT, and restart brief untouched

## Anti-goals

- do not rewrite the whole README
- do not hide features that genuinely exist
- do not reopen product-boundary debate in this slice
- do not widen into code changes
- do not use docs cleanup as a pretext for roadmap reshuffling

## Notes for implementation

- Keep the slice boring.
- Classification clarity matters more than selling polish.
- If a section makes execution or analytics feel co-equal with prompt asset management, tone it down.