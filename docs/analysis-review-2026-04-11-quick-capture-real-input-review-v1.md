# PromptManager — Analysis Review

Date: 2026-04-11
Target: `Quick Capture Real-Input Review v1`
Expected source: `docs/analysis-brief-2026-04-11-quick-capture-real-input-review-v1.md`
Reviewer: main

## Verdict

**No new slice recommended yet.**

The current Quick Capture cleanup seam looks appropriately bounded and already covers the most credible high-frequency wrapper noise classes visible in the present representative sample:
- outer fenced block,
- outer prompt label,
- outer blockquote,
- unchanged mixed or transcript-like input.

At this point, the stronger move is to let the current seam absorb more real usage before introducing another cleanup rule.

## Review basis

This pass used a **small representative sample**, not production telemetry.

Sources reviewed:
- `gui/dialogs/quick_capture.py`
- `tests/test_quick_capture_dialog.py`
- the delivered briefs for:
  - `Fence Unwrap v1`
  - `Prompt Label Strip v1`
  - `Blockquote Unwrap v1`

Representative sample shapes covered by the current seam and tests:
1. plain prompt body
2. outer fenced markdown block
3. outer single-line blockquote
4. outer multiline blockquote
5. outer prompt label inline
6. outer prompt label multiline
7. ambiguous bare label
8. transcript-like follow-on content after a label
9. mixed prose plus fenced block
10. mixed quoted and non-quoted lines
11. effectively empty quoted wrapper
12. title-derived messy input with quote markers in the body

This is enough for a bounded review of direction, but not enough to justify a new cleanup rule with confidence.

## What looks good already

### 1. The cleanup seam is still boring in the right way
`QuickCaptureDraft.to_prompt()` currently applies a short deterministic chain:
1. blockquote unwrap
2. fence unwrap
3. prompt-label strip

That is still understandable, local, and reversible.
It does not yet look like a parser framework.

### 2. The current slices cover the most credible wrapper-noise classes
The present seam already handles the three most believable real-world outer wrappers for copied prompt text:
- markdown fence
- `Prompt:` / `User prompt:` / `System prompt:` label
- markdown blockquote

That is a good practical baseline for messy copy-paste input.

### 3. Guardrails against over-cleanup are healthy
The current behavior intentionally refuses to clean up:
- mixed quoted and unquoted content
- transcript-like role content after prompt labels
- incomplete or effectively empty wrappers
- mixed prose plus fenced block content

Those refusals are good.
They reduce the chance that Quick Capture starts mutating ambiguous content aggressively.

## Most common input-shape conclusion

From the current representative sample, the strongest recurring shapes are:
- plain prompt input,
- one obvious outer wrapper,
- messy but clearly non-automatable mixed input.

That split matters.
It suggests the current cleanup seam already covers the high-value obvious wrappers while deliberately leaving the ambiguous class alone.

## What is still missing

### 1. Real-world frequency data
The current review does **not** prove which messy shape is most common in day-to-day operator use.
It only shows that the current cleanup rules map well to believable prompt-copy patterns.

### 2. Strong evidence for one more bounded cleanup rule
No remaining miss stands out strongly enough from the present sample to justify immediate implementation.

Potential future candidates exist, but none is justified yet from the available evidence:
- stacked wrapper-specific handling beyond what the current chain already incidentally supports
- header/title noise cleanup ahead of the prompt body
- broader transcript extraction
- mixed prose plus prompt extraction

Right now those are tempting, not earned.

## Recommendation

**Keep the current Quick Capture behavior unchanged for now.**

Recommended next operational move:
- let the existing seam collect more real usage,
- keep noticing where cleanup still fails in practice,
- only open a new slice when one miss pattern clearly repeats and can be described as one boring deterministic rule.

## If a future next slice becomes justified

The strongest future candidate area is likely:
- **wrapper plus title/header noise**

But only if repeated real captures show the same pattern and the fix can stay local to `QuickCaptureDraft.to_prompt()` without turning into document parsing.

That is not yet a recommendation to build it.
It is only the most plausible next watch area.

## Anti-recommendations

Still do **not** do any of the following from this review:
- do not build transcript parsing
- do not extract prompts from mixed prose automatically
- do not add broad markdown cleanup
- do not create import modes or cleanup settings
- do not turn incidental stacked-wrapper support into a general wrapper engine
- do not force a new slice just because the last three slices were successful

## Decision summary

Result:
- **no new slice recommended yet**

Reason:
- the current Quick Capture cleanup seam already covers the most credible obvious wrapper-noise cases in the available representative sample,
- and no remaining miss pattern is yet strong enough to justify another bounded rule without drifting toward parser behavior.
