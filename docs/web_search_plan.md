# Web Search Integration Plan

## Research Trace

- **Exa Quickstart (2025-12-04)** — [docs.exa.ai/reference/quickstart](https://docs.exa.ai/reference/quickstart) shows that the official Python SDK (`exa-py`) exposes `search`, `search_and_contents`, and `answer` helpers once an `EXA_API_KEY` environment variable is provided. Exa also supports raw HTTP calls, but the SDK reduces request boilerplate and handles result pagination in one call.
- **Exa Python Examples** (via `exa_py` README snippets surfaced through Exa-code search) demonstrate a consistent parameter surface (`type="auto"`, `include_domains`, `text_contents_options`, etc.) that maps cleanly onto PromptManager's structured search needs. Highlight support and similarity lookups arrive through the same client, so an abstraction that wraps Exa's client objects will cover most future workflows.
- **Tavily Search API Reference (2025-12-07)** — [docs.tavily.com/documentation/api-reference/endpoint/search](https://docs.tavily.com/documentation/api-reference/endpoint/search) details the `/search` endpoint, including bearer authentication via `tvly-` keys, payload parameters (`query`, `max_results`, `topic`, `search_depth`, `include_domains`, `include_answer`, etc.), and JSON responses that return `answer`, `results[{title,url,content,score}]`, optional raw content, favicons, and timing metadata.
- **Serper Search API (2025-12-07)** — Elastic’s [“Using AutoGen with Elasticsearch”](https://www.elastic.co/search-labs/blog/using-autogen-with-elasticsearch) tutorial and Rui Ramos’ [Serper quickstart](https://rramos.github.io/2024/06/13/serper/) both demonstrate POST requests to `https://google.serper.dev/search` with an `X-API-KEY` header plus JSON payload (`q`, `num`, optional `gl`/`hl`/`location`), returning an `organic` array with `title`, `link`, `snippet`, `date`, and `source` for each document.
- **SerpApi Google Search API (2025-12-07)** — [serpapi.com/search-api](https://serpapi.com/search-api) documents the GET `/search` endpoint that accepts `api_key`, `q`, `num`, optional localization filters (`location`, `gl`, `hl`, `google_domain`, `uule`), advanced switches (`tbs`, `tbm`, `safe`, `device`, `start`), and returns JSON with `organic_results[{title,link,snippet,snippet_highlighted_words,date,source}]`, local results, shopping blocks, and pagination metadata.
- **Google Programmable Search (2025-12-07)** — [developers.google.com/custom-search/v1/using_rest](https://developers.google.com/custom-search/v1/using_rest) explains how to issue GET `https://www.googleapis.com/customsearch/v1` requests with a `key` (API key) and `cx` (Programmable Search Engine ID) alongside query parameters (`q`, `num`, `start`, `safe`, `lr`, etc.), returning an `items[{title,link,snippet,htmlSnippet,displayLink,pagemap}]` array plus `searchInformation`.

## Objectives

1. **Immediate:** Let operators enter Exa, Tavily, Serper, SerpApi, or Google Programmable Search credentials inside the PromptManager settings dialog so future workflows can dispatch authenticated web searches without touching env vars.
2. **Near-term:** Introduce a provider-agnostic web search service that can:
   - Register multiple providers (Exa + Tavily + Serper + SerpApi to start).
   - Normalize search queries (`query`, `limit`, `filters`) and return a stable `WebSearchResult` payload for UI/automation layers.
   - Handle provider capability discovery (e.g., answer vs. similarity) without leaking implementation details to callers.
3. **Future:** Wire prompt authoring/execution surfaces so they can:
   - Trigger background research runs (“Search latest guidance”) and annotate prompts with citations.
   - Enrich diagnostics (lint/validation, execution troubleshooting) with contextual web snippets.

## Phased Work Breakdown

### Phase 1 – Configuration & Secrets

- Extend `PromptManagerSettings` with:
- `web_search_provider` (`Literal["exa", "tavily", "serper", "serpapi", "google", "random"]`, default `exa` until telemetry indicates otherwise). `random` rotates between providers that have API keys configured, falling back to the only available provider when necessary.
  - Secret fields for provider credentials (e.g., `exa_api_key`, `tavily_api_key`, `serper_api_key`, `serpapi_api_key`, `google_api_key`, `google_cse_id`, stored only in memory and env vars such as `EXA_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`, `SERPAPI_API_KEY`, `GOOGLE_API_KEY`, or `GOOGLE_CSE_ID`).
- Update CLI summaries, runtime persistence, and docs (`README`, `README-DEV`, env matrices) so every provider shows up alongside LiteLLM.
- Settings dialog: add an **Integrations → Web Search** tab hosting the provider selector plus per-provider API key password fields. Keys should stay in memory only (mirroring LiteLLM).

### Phase 2 – Provider Architecture

- Add `core/web_search` package with:
  - `WebSearchProvider` protocol + `WebSearchResult` / `WebSearchDocument` dataclasses.
  - Provider implementations for Exa, Tavily, Serper, and SerpApi using `httpx.AsyncClient`.
  - `WebSearchService` that resolves the configured provider, exposes async `search()` plus a sync wrapper for CLI/tests.
- Wire `build_prompt_manager` to construct a `WebSearchService` (if an API key + provider is available) so PromptManager exposes `self.web_search`.

### Phase 3 – UX & Workflow Hooks (Future Iterations)

- **Prompt Authoring:** add “Research with web search” action to prompt editor/workbench that fetches snippets relevant to the current goal/context and stores citations with the prompt.
- **Execution History:** add troubleshooting helper (“Suggest fix using web search”) that queries the provider when executions fail and surfaces top remediation steps/toast notifications.
- **Automation:** extend CLI to offer `python -m main web-search "<query>" --provider exa --limit 5` for scripting; integrate with diagnostics so offline workflows stay parity.

### Operational Considerations

- **Rate Limits:** Exa’s SDK defaults to `type="auto"`; expose toggles for keyword or neural search later if telemetry warrants.
- **Error Handling:** wrap provider errors with actionable hints (missing API key, HTTP 4xx, quota). Retry with exponential backoff using `tenacity` if the provider signals a transient failure.
- **Testing:** rely on `respx` to mock Exa responses in strict mode; property-based tests can ensure search results preserve ordering and metadata.
- **Security:** never persist provider secrets to disk; mask values in CLI output and logs (reuse `mask_secret` helper).
