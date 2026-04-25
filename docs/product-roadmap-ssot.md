# PromptManager — Product Roadmap SSOT

Status: active
Updated: 2026-04-25
Related canonical file: `docs/product-ssot.md`

## Purpose

This file translates the product SSOT into a concise development sequence.
It is intentionally ordered by dependency and product leverage, not by calendar.

If this file conflicts with `docs/product-ssot.md`, the product SSOT wins.

---

## Roadmap principle

PromptManager should develop in this order:

**trust surfaces first, prompt asset loop second, operational leverage third, expansion last**

This means:
1. make configuration and runtime state legible
2. make the core prompt loop faster and more reliable
3. make execution and comparison useful in service of prompt assets
4. expand outward only after the center is coherent

---

## Stage 1 — Trust and operability foundation

Focus:
- deterministic settings resolution
- readable effective config
- clear `OK` / `WARN` / `FAIL` diagnostics
- visible routing and embedding decisions
- fail-fast behavior for invalid critical runtime state
- consistent GUI/CLI status language

Why first:
- users should understand whether PromptManager is ready before they trust anything else
- routing, embeddings, API auth, and cache state are product trust surfaces, not implementation trivia

Done looks like:
- GUI and CLI expose the same core operational truth
- users can see what is configured, what is optional, and what blocks real use
- misconfiguration stops being silent

---

## Stage 2 — Core prompt asset loop quality

Focus:
- faster capture
- clearer draft-to-asset promotion
- better duplicate/similarity signals
- stronger recent/search/retrieve flows
- more readable inspect/details view
- lower-friction reuse, copy, export, and refine actions

Why second:
- this is the product center
- if capture/retrieve/inspect/reuse is weak, no ops layer can compensate

Done looks like:
- prompts move cleanly from rough input to reusable asset
- users can reliably find and judge assets later
- reuse becomes faster than rediscovery

---

## Stage 3 — Trustworthy prompt operations

Focus:
- bounded prompt execution
- recorded runs linked to assets
- simple baseline vs candidate comparisons
- run provenance and lightweight history
- refinement supported by visible evidence

Why third:
- execution matters when it helps decide whether a prompt should be reused, refined, forked, or retired
- operations should strengthen the asset library, not distract from it

Done looks like:
- runs are easy to understand and compare
- execution produces usable decision support, not noise
- prompt refinement becomes evidence-backed

---

## Stage 4 — Controlled automation surfaces

Focus:
- stronger CLI workflows
- scriptable bounded run flows
- repeatable validation paths
- automation that respects the same settings and diagnostics model

Why fourth:
- automation is valuable only after the core model is stable and legible
- hidden or parallel behavior would weaken trust

Done looks like:
- automation extends the existing product model instead of creating a second one
- headless use remains understandable and debuggable

---

## Stage 5 — Selective expansion

Focus:
- secondary sharing features
- optional collaboration surfaces
- bounded analytics improvements
- convenience layers that do not blur the product center

Why last:
- breadth before coherence creates noise
- PromptManager should earn expansion by first becoming sharp and trustworthy

Done looks like:
- new surfaces feel additive, not identity-changing
- the product still reads clearly as a local-first home for prompt assets with an operational layer

---

## Priority filters for any new work

Before prioritizing a feature, ask:
1. Does it improve trust, clarity, or determinism?
2. Does it strengthen capture, retrieve, inspect, reuse, or refine?
3. Does it help users make better prompt decisions?
4. Does it preserve a single coherent product model?
5. Is it more valuable than improving an already-visible weak spot?

If mostly no, it should not outrank current core work.

---

## Explicit de-prioritization rule

Do not prioritize these ahead of the roadmap above unless directly required by the core loop:
- broad dashboarding
- novelty assistant behavior
- large workbench expansion
- voice-first flows
- heavy scenario generation investment
- collaboration-first redesign
- automation that introduces opaque background behavior

---

## Operational takeaway

If product direction feels ambiguous, the decision rule is:

**make PromptManager more legible, more trustworthy, and better at managing prompt assets before making it broader.**
