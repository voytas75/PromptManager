# PromptManager Prompt Chain Clarity / Continuity Roadmap

Status: proposed

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Improve operator trust and readability of the existing prompt-chain workflow so chain input, step-to-step handoff, and final output are easier to understand without turning PromptManager into a broader orchestration product.

**Architecture:** This roadmap stays deliberately bounded on the existing prompt-chain seams already present in `cli/commands.py`, `cli/parser.py`, `core/prompt_manager/chains.py`, and the current GUI chain dialog/editor surfaces under `gui/dialogs/`. The cycle should prefer small continuity cues, calmer run summaries, and guard coverage over any engine expansion. No new branching runtime, workflow language, persistence model, or chain-first product framing should be introduced.

**Tech Stack:** Python 3.13, PySide6 widgets, existing PromptManager chain core/CLI seams, pytest, Ruff, Pyright, SSOT and roadmap docs under `docs/`.

---

## Why this roadmap exists

Live repo review confirms:
- Prompt chains already exist across GUI, CLI, models, backend mixins, and tests, so this is not a greenfield feature area,
- `docs/product-ssot.md` and `docs/product-boundary-ssot.md` explicitly reject **heavy prompt-chain expansion** as a preferred direction,
- recent product positioning work consistently treats chains as a **secondary/supporting surface**, not the main product center,
- the repo already contains bounded chain seams (`prompt-chain-list`, `prompt-chain-show`, `prompt-chain-apply`, `prompt-chain-run`, GUI manager/editor/run panel),
- the highest-value next move is therefore not “more chain power”, but making the existing chain path calmer, more legible, and easier to trust when an operator actually uses it.

Short version:

**keep prompt assets as the center; make existing chains easier to read and safer to interpret without expanding them into workflow automation.**

---

## Product intent for this cycle

This cycle should answer:

> given that prompt chains already exist as a bounded supporting surface, what is the smallest next set of clarity and continuity improvements that helps operators understand what a chain consumes, what each step passes forward, and what the final result means?

This roadmap strengthens three things:

1. **input clarity** — the operator should immediately understand what entered the chain,
2. **step continuity** — the operator should not need to infer how one step’s output becomes the next step’s input,
3. **bounded interpretation trust** — chain results should read as calm execution evidence, not as a new orchestration system.

---

## Constraints

Do not add:
- branching or conditional workflow expansion,
- multi-path routing or retries-as-engine semantics,
- new chain storage abstractions or schema systems,
- hidden background execution layers,
- chain analytics expansion as a parallel product surface,
- chain-first wording that competes with the prompt asset center,
- broad CLI/API redesign outside the existing chain seams.

Prefer first:
- `gui/dialogs/prompt_chains.py`
- `gui/dialogs/prompt_chain_editor.py`
- `core/prompt_manager/chains.py`
- `cli/commands.py`
- focused regressions in `tests/test_prompt_chain_dialog.py`, `tests/test_prompt_chain_backend.py`, and nearby chain CLI tests if added.

---

## Confirmed baseline before this cycle

Assume already delivered and do not reopen without a failing regression:
- prompt chain CRUD already exists in GUI,
- prompt chain CLI already exists for list/show/apply/run,
- chain execution already supports a single plain-text input flowing across ordered steps,
- current chain surfaces already support web-search toggle wiring and bounded result rendering,
- test coverage already exists for chain model construction, backend behavior, dialog rendering, and parts of chain run presentation.

This roadmap should build on those seams instead of widening the chain feature set.

---

## Recommended execution shape

Work one bounded slice at a time:
1. add one RED test at the nearest chain seam,
2. confirm the exact failure,
3. implement the smallest clarity/continuity change that resolves it,
4. run targeted chain tests,
5. update this roadmap and `docs/CHANGELOG.md`,
6. commit/push before starting the next slice.

If the first probe already passes, do **not** force a runtime/UI change just to justify the roadmap. Close that slice as guard-only with explicit evidence.

---

## Slice candidates

### Slice 1 — Prompt chain run handoff clarity v1

**Status:** done (2026-04-29)

**Delivered:**
- chain run results now render explicit per-step input continuity for later successful steps,
- step 1 keeps `Input to step`, while step 2+ now shows `Input from previous step output`,
- the slice stayed local to the existing GUI run-result presentation seam in `gui/dialogs/prompt_chains.py`,
- focused dialog coverage now locks the later-step handoff cue without changing chain execution logic or persistence.

**Boundaries:**
- no engine behavior changes,
- no variable-schema comeback,
- no branching/condition redesign.

### Slice 2 — Prompt chain final-output emphasis v1

**Status:** proposed

**Intent:**
- make the terminal result of a chain easier to distinguish from intermediate step output,
- reduce ambiguity about which text is the chain’s end result,
- keep the change limited to labels, grouping, or rendering cues on existing result surfaces.

**Boundaries:**
- no analytics expansion,
- no export workflow redesign,
- no extra persistence.

### Slice 3 — Prompt chain neutral empty-state / reset guard pack v1

**Status:** proposed

**Intent:**
- verify that clearing or re-running chain inputs does not leave stale summary/result cues behind,
- ensure chain run surfaces return to a neutral, understandable state after reset paths,
- allow this slice to close as guard-only if the current runtime already behaves correctly.

**Boundaries:**
- no new run-history model,
- no lifecycle redesign,
- no hidden state store.

### Slice 4 — Prompt chain CLI output legibility v1

**Status:** proposed

**Intent:**
- tighten `prompt-chain-show` and/or `prompt-chain-run` output so console operators can read the chain structure and result flow with less ambiguity,
- keep the work read-first and deterministic,
- prefer formatting clarity over feature breadth.

**Boundaries:**
- no new API layer,
- no machine-contract redesign unless a tiny existing output tweak clearly justifies it,
- no separate export subsystem.

---

## Recommended first slice

Start with **Slice 1 — Prompt chain run handoff clarity v1**.

Why this first:
- it is the most directly user-facing continuity gap in the current chain experience,
- it improves trust without expanding chain power,
- it creates a cleaner base for later final-output emphasis or reset guards,
- it can likely stay on an existing dialog/backend presentation seam with focused tests.

---

## Implementation brief for the next slice

Create a short brief only if recon shows the first slice needs coordinated changes across backend result models, GUI rendering, and CLI output at once. If the needed work stays small and local to one run/result seam plus focused tests, this roadmap can serve as the brief.

---

## Success criteria for the whole roadmap

This cycle is successful if, by the end:
- operators can tell what input entered a chain and what result came out more quickly,
- step-to-step flow is easier to understand without reading code or guessing hidden behavior,
- chain surfaces feel calmer and more trustworthy,
- PromptManager still reads as a prompt-asset product with chains as a bounded supporting surface rather than a new automation center.
