# PromptManager — Analysis Brief

Date: 2026-04-11
Status: ready
Focus: Quick Capture Real-Input Review v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-10-fence-unwrap-v1.md`
- `docs/implementation-brief-2026-04-11-prompt-label-strip-v1.md`
- `docs/implementation-brief-2026-04-11-blockquote-unwrap-v1.md`

## Goal

Run one bounded analysis pass on **real Quick Capture input shapes** so the next cleanup or normalization slice is chosen from actual usage patterns rather than speculation.

## Why this now

Quick Capture now has three small deterministic cleanup passes:
- `Fence Unwrap v1`
- `Prompt Label Strip v1`
- `Blockquote Unwrap v1`

That is enough local behavior to justify a reality check.

The point of this review is not to redesign capture.
The point is to answer one practical question:

> what messy input shape appears often enough, and hurts enough, to deserve the next bounded cleanup slice?

## Product intent

This analysis supports the core loop at:
- capture,
- normalize,
- reuse.

It should improve confidence that future Quick Capture changes reflect real operator friction instead of hypothetical cleanup ideas.

## Scope

### In scope
- review a bounded sample of real or representative Quick Capture input examples
- classify common raw-input shapes
- check where current cleanup already works well
- identify the most valuable remaining false negatives
- identify any risky false positives or over-cleanup patterns
- recommend exactly one next bounded slice only if evidence supports it

### Out of scope
- implementing the next slice in this pass
- broad import-pipeline design
- transcript parsing redesign
- generic markdown parser work
- schema or storage changes
- UI redesign
- roadmap generation or multi-slice planning

## Recommended sample shape

Use a small bounded sample, for example:
- 15 to 30 real prompt captures, or
- a smaller curated set if that is what is available now

Prefer examples that represent actual messy input sources such as:
- copied chat prompts
- notes or docs snippets
- ticket or incident writeups
- copied quoted prompts
- fenced prompt blocks
- pasted prompt wrappers with labels

## Review method

For each captured example, record only what matters:
1. raw input shape
2. intended usable prompt body
3. whether current Quick Capture cleanup already produces the right result
4. if not, what kind of miss it is:
   - acceptable false negative
   - painful false negative
   - dangerous false positive
5. whether the miss appears repeatable enough to justify a bounded new slice

## Suggested classification buckets

Keep the buckets boring and practical.

Examples:
- plain prompt, no cleanup needed
- outer fenced block
- outer prompt label
- outer blockquote
- mixed prose + prompt
- transcript-like input
- stacked wrappers
- wrapper plus title/header noise
- unclear / not worth automating

Do not create taxonomy theatre.
The buckets only exist to help choose the next real slice.

## Output expected from this review

The review should produce:
1. a short list of the most common input shapes
2. a short statement of where current cleanup is already good enough
3. one clear recommendation:
   - no new slice yet, or
   - one next bounded cleanup slice
4. a short anti-recommendation list of tempting ideas that should still stay out of scope

## Decision rule

A new Quick Capture slice is justified only if all of the following are true:
1. the raw-input pattern appears more than once or is clearly recurring
2. the current miss creates real cleanup friction
3. the fix can stay local to the Quick Capture conversion seam
4. the first version can be tested deterministically
5. the change can be described as one boring cleanup rule, not a parsing system

If those conditions are not met, stop at the analysis and keep the current behavior.

## Acceptance checks

1. The review uses a bounded sample instead of open-ended repo or history mining.
2. The findings distinguish clearly between acceptable misses and worthwhile misses.
3. The output recommends at most one next slice.
4. The output includes explicit anti-goals to prevent parser drift.
5. If no strong candidate emerges, the review explicitly says so.

## Rollback

Rollback is trivial because this pass is analysis-only:
- discard the recommendation
- keep current Quick Capture behavior unchanged
- do not force implementation just because a review was written

## Anti-goals

- do not turn this into a backlog workshop
- do not widen into import architecture
- do not treat every messy input shape as automation-worthy
- do not justify parser complexity with isolated edge cases
- do not bundle implementation into the review pass

## Suggested next step after this brief

If Wojtek wants to proceed, do one bounded review artifact such as:
- `docs/analysis-review-2026-04-11-quick-capture-real-input-review-v1.md`

That follow-up should end with exactly one of these outcomes:
- `no new slice recommended yet`
- `recommended next slice: <one bounded cleanup slice>`
