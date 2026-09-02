# PromptManager Prompt-Library Search / Filter Continuity Roadmap

Status: done

> **For Hermes:** Use subagent-driven-development skill to implement this roadmap one bounded slice at a time.

**Goal:** Strengthen prompt-library retrieval trust by making active search and filter narrowing more explicit at the existing filter-panel seam, so operators immediately understand when the visible library is narrowed and why.

**Architecture:** This roadmap stays in Stage 2 and reuses the current prompt-library filter surface rather than redesigning search, ranking, or persistence. Each slice should remain local to the existing `PromptFilterPanel` / nearby prompt-list seams, prefer visible continuity cues and guard coverage, and avoid introducing new storage, a second retrieval model, or broader workflow semantics.

**Tech Stack:** Python 3.13, PySide6 widgets, existing prompt-list/filter panel seams, pytest, Ruff, Pyright, product docs under `docs/`.

---

## Why this roadmap exists

Live repo review confirms:
- the just-finished CLI / API console-access cycle is now closed, so the most valuable next work should return to the product center instead of extending Stage 4 by inertia,
- `docs/product-ssot.md` prioritizes core prompt asset loop quality, especially stronger search / retrieve flows and lower-friction reuse,
- recent Stage 2 cycles improved entry points (`Quick Capture`, `Recent Prompts`, `Promote Draft`), favorites clarity, and prompt actions clarity,
- the prompt-library filter panel already carries multiple bounded operator cues (`Tag filter: ...`, `Favorites filter: ...`, `Sort locked during search`), which makes it a proven seam for small trust-building retrieval improvements,
- the next useful move is therefore not a search redesign, new ranking system, or another automation surface, but one fresh bounded cycle that makes narrowing state easier to read and harder to misinterpret.

Short version:

**capture / retrieve / inspect / reuse stays the center — make narrowing state calmer and clearer before broadening the product again.**

---

## Product intent for this cycle

This cycle should answer:

> when the prompt library is already narrowed by active search and local filters, what is the smallest next set of cues/guards that makes that narrowed state obvious enough that operators trust what they are seeing without needing to infer hidden rules?

This roadmap strengthens three things:

1. **visible narrowing state** — the operator should immediately understand whether the library is broad or constrained,
2. **continuity between filter controls and list meaning** — local controls should explain their effect without requiring a second status surface,
3. **single product model** — all work should stay on the existing filter/list seams and avoid new retrieval abstractions.

---

## Constraints

Do not add:
- a search redesign,
- new ranking logic,
- saved filter sets or new persistence,
- background indexing or asynchronous retrieval,
- CLI/headless parity work unless a cue actually starts traveling through shared fields,
- extra dashboard/status surfaces outside the existing prompt-library seams.

Prefer first:
- `gui/widgets/prompt_filter_panel.py`
- nearby prompt-list presenter/coordinator seams only when the filter-panel seam is truly insufficient,
- focused regression coverage in `tests/test_prompt_filter_panel.py` and adjacent prompt-list tests,
- changelog + roadmap updates only after a bounded slice is actually green.

---

## Confirmed baseline before this cycle

Assume already delivered and do not reopen without a failing regression:
- tag-filter helper visibility (`Tag filter: all tags` / active tag wording),
- favorites-only helper visibility (`Favorites filter: all prompts` / `favorites only`),
- search-lock cue on the disabled sort control (`Sort locked during search`),
- GUI-local parity guards proving those retrieval cues do not leak into shared analytics/headless fields,
- existing prompt-list retrieval/discovery wording from earlier Stage 2 cycles.

This roadmap should build on those seams instead of relitigating already-shipped cues.

---

## Recommended execution shape

Work one bounded slice at a time:
1. add one RED test at the nearest filter/search seam,
2. confirm the exact failure,
3. implement the smallest visible/runtime change or guard-only closure,
4. run targeted prompt-filter/prompt-list smoke,
5. update this roadmap and `docs/CHANGELOG.md`,
6. commit/push before picking the next slice.

If the first probe already passes, do **not** force a runtime change just to justify the roadmap. Close the slice as guard-only or `covered by existing behavior`, depending on the evidence.

---

## Slice candidates

### Slice 1 — Active narrowing summary cue v1

**Status:** done (2026-04-29)

**Delivered:**
- added one always-visible `activeNarrowingSummaryLabel` on the existing `PromptFilterPanel` seam,
- summary now flips from `Showing all prompts` to a compact combined narrowing cue when search/filter constraints are active,
- first v1 coverage locks the combined search + tag + favorites case without changing ranking, persistence, or broader prompt-list semantics.

**Notes:**
- This slice stays entirely on the filter-panel seam.
- Search wording is currently driven by the existing sort-lock/search-active state and remains deliberately bounded for v1.

### Slice 2 — Search-query continuity cue v1

**Status:** done (2026-04-29)

**Delivered:**
- active narrowing summary now uses the live search query text instead of a generic search-active placeholder,
- clearing search cleanly restores the neutral summary (`Showing all prompts`),
- implementation stays on the existing `PromptSearchController` + `PromptFilterPanel` seam without changing ranking, persistence, or CLI/headless behavior.

**Notes:**
- Query continuity is intentionally bounded to the local filter/search seam.
- The summary remains calm and deterministic: live query when search is active, no search fragment when cleared.

### Slice 3 — Neutral reset-state guard pack v1

**Status:** implemented (2026-04-29, guard-only)

**Delivered:**
- added one focused regression guard proving the active narrowing summary returns to `Showing all prompts` after search, tag, and favorites constraints are fully cleared,
- verified the existing runtime seam already resets cleanly, so no production-code change was required,
- kept the slice local to `tests/test_prompt_filter_panel.py` without widening into presenter/list logic or persistence.

**Notes:**
- This slice intentionally closes as guard-only because the first probe passed immediately.
- Neutral reset behavior is now explicitly locked for the combined search + filters path.

---

## Recommended first slice

Start with **Slice 1 — Active narrowing summary cue v1**.

Why this first:
- it has the highest leverage relative to current filter-panel maturity,
- it can unify the operator reading of active search + filter state without widening scope,
- it is more directly user-facing than a pure guard pack,
- it creates a cleaner base for later search-query continuity and reset-state guards.

---

## Implementation brief for the next slice

Create a short brief only if recon shows the summary cue needs coordination beyond `PromptFilterPanel` itself. If the slice fits inside the existing panel seam with one small helper label/copy change and focused tests, the roadmap can serve as the brief.

---

## Success criteria for the whole roadmap

This cycle is successful if, by the end:
- operators can tell more quickly when the prompt library is narrowed and why,
- local search/filter cues stay calm, deterministic, and easy to reset,
- retrieval trust improves without introducing a second retrieval model or broader workflow complexity,
- PromptManager remains centered on prompt assets rather than expanding into a larger search product.
