# PromptManager — `prompt-find` Semantic Ranking Repair v1

Status: completed
Owner: PromptManager Team

## Goal

Restore `prompt-find` as a predictable natural-language semantic search: result order must follow the supplied query, not intent-classifier or user-profile recommendation bias.

## Confirmed diagnosis

- `prompt-find` currently calls `PromptManager.suggest_prompts()`.
- `suggest_prompts()` augments the embedding query with heuristic classifier category/tag hints, reorders candidates with `rank_by_hints()`, and applies user-profile personalization.
- For `analyze video`, the classifier creates `Code Analysis` / `analysis, tests, review` hints. A raw video candidate is then reordered behind Code Analysis candidates and profile-favoured tags.
- The Windows catalog and Chroma index are populated: 329 prompts and 329 stored embeddings, with a 3072-dimensional cosine collection. The observed defect is post-retrieval ranking policy, not missing catalog vectors.

## Bounded decision

- `prompt-find` will call the existing raw semantic domain seam, `search_prompts(query, ...)`, with the original trimmed user query.
- `prompt-find` will preserve Chroma semantic order; explicit CLI metadata filters remain post-ranking filters.
- `suggest` and GUI suggestions retain `suggest_prompts()` and its intent/personalization behavior as a separate recommendation surface.
- No embedding rebuild, provider/configuration change, index schema change, or broader recommendation-system rewrite is included.

## Execution order

1. Add RED CLI and domain regressions proving raw semantic order survives `prompt-find` and profile bias does not change it.
2. Route `prompt-find` through `search_prompts()` without classifier/query augmentation.
3. Preserve filters and output contracts; update the test double accordingly.
4. Update CLI operational documentation and changelog with the search-versus-suggest boundary.
5. Run focused tests, isolated deterministic CLI smoke, Ruff, format, CI-scope Pyright, and diff hygiene.

## Acceptance criteria

- A raw semantic order `[video, code]` remains `[video, code]` through `prompt-find`.
- `prompt-find` forwards exactly `analyze video` to semantic search, not a classifier-augmented string.
- Existing `suggest` behavior stays covered and unchanged.
- `prompt-find` text/JSON/filter behavior remains intact.

## Completion update

Completed 2026-09-20:

- Routed `prompt-find` directly through `PromptManager.search_prompts(query, limit=...)` using the original trimmed natural-language query.
- Preserved raw semantic order and retained category/tag/source/active filtering only after semantic retrieval.
- Kept `suggest` and GUI suggestion paths on `suggest_prompts()`, including their intent-hinted and personalized recommendation behavior.
- Added a CLI regression that asserts `analyze video` reaches raw search unchanged and keeps a video result ahead of a Code Analysis result.
- Added a domain regression that locks the raw semantic order and similarity values without classifier or profile-bias reordering.
- Updated parser/help wording, the developer CLI index, and changelog to make the search-versus-suggest boundary explicit.

Verification completed:

- focused CLI `prompt-find` tests: `6 passed`;
- focused domain search/suggestion tests: `7 passed`;
- selected docs/CLI/domain regressions: `144 passed`;
- Ruff lint and format: passed;
- Pyright for changed tests: `0 errors`;
- CI-scope Pyright (`main.py config models`): `0 errors`;
- `git diff --check`: passed.

Live production-catalog acceptance remains **to verify on Windows**: the Windows catalog's Azure embedding query is a live provider boundary. The local deterministic smoke confirms that this repair stops the application-level reordering, but it cannot prove the Azure embedding model itself ranks the production records as expected. Re-run the two user queries from `D:\repos\PromptManager` after this change is synchronized and compare the ordered results.
