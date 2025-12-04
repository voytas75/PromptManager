# Web Search Integration Plan

## Research Trace

- **Exa Quickstart (2025-12-04)** — [docs.exa.ai/reference/quickstart](https://docs.exa.ai/reference/quickstart) shows that the official Python SDK (`exa-py`) exposes `search`, `search_and_contents`, and `answer` helpers once an `EXA_API_KEY` environment variable is provided. Exa also supports raw HTTP calls, but the SDK reduces request boilerplate and handles result pagination in one call.
- **Exa Python Examples** (via `exa_py` README snippets surfaced through Exa-code search) demonstrate a consistent parameter surface (`type="auto"`, `include_domains`, `text_contents_options`, etc.) that maps cleanly onto PromptManager's structured search needs. Highlight support and similarity lookups arrive through the same client, so an abstraction that wraps Exa's client objects will cover most future workflows.

## Objectives

1. **Immediate:** Let operators enter an Exa API key inside the PromptManager settings dialog so future workflows can dispatch authenticated web searches without touching env vars.
2. **Near-term:** Introduce a provider-agnostic web search service that can:
   - Register multiple providers (start with Exa).
   - Normalize search queries (`query`, `limit`, `filters`) and return a stable `WebSearchResult` payload for UI/automation layers.
   - Handle provider capability discovery (e.g., answer vs. similarity) without leaking implementation details to callers.
3. **Future:** Wire prompt authoring/execution surfaces so they can:
   - Trigger background research runs (“Search latest guidance”) and annotate prompts with citations.
   - Enrich diagnostics (lint/validation, execution troubleshooting) with contextual web snippets.

## Phased Work Breakdown

### Phase 1 – Configuration & Secrets

- Extend `PromptManagerSettings` with:
  - `web_search_provider` (`Literal["exa"]` to start, default `exa`).
  - Secret fields for provider credentials (e.g., `exa_api_key`, stored only in memory and env vars).
- Update CLI summaries, runtime persistence, and docs (`README`, `README-DEV`, env matrices) so the new provider shows up alongside LiteLLM.
- Settings dialog: add an **Integrations → Web Search** tab hosting the provider selector and Exa API key password field. Keys should stay in memory only (mirroring LiteLLM).

### Phase 2 – Provider Architecture

- Add `core/web_search` package with:
  - `WebSearchProvider` protocol + `WebSearchResult` / `WebSearchDocument` dataclasses.
  - `WebSearchRegistry` to register providers at import time.
  - `WebSearchService` that resolves the configured provider, exposes async `search()` plus a sync wrapper for CLI/tests.
  - `ExaWebSearchProvider` implementation that depends on `httpx.AsyncClient` (lightweight) and supports core params (`query`, `num_results`, `text`, `summary`, `livecrawl`).
- Wire `build_prompt_manager` to construct a `WebSearchService` (if an API key + provider is available) so PromptManager exposes `self.web_search`.

### Phase 3 – UX & Workflow Hooks (Future Iterations)

- **Prompt Authoring:** add “Research with Exa” action to prompt editor/workbench that fetches snippets relevant to the current goal/context and stores citations with the prompt.
- **Execution History:** add troubleshooting helper (“Suggest fix using web search”) that queries the provider when executions fail and surfaces top remediation steps/toast notifications.
- **Automation:** extend CLI to offer `python -m main web-search "<query>" --provider exa --limit 5` for scripting; integrate with diagnostics so offline workflows stay parity.

### Operational Considerations

- **Rate Limits:** Exa’s SDK defaults to `type="auto"`; expose toggles for keyword or neural search later if telemetry warrants.
- **Error Handling:** wrap provider errors with actionable hints (missing API key, HTTP 4xx, quota). Retry with exponential backoff using `tenacity` if the provider signals a transient failure.
- **Testing:** rely on `respx` to mock Exa responses in strict mode; property-based tests can ensure search results preserve ordering and metadata.
- **Security:** never persist provider secrets to disk; mask values in CLI output and logs (reuse `mask_secret` helper).
