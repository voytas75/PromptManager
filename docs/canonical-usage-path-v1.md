# PromptManager — Canonical Usage Path v1

Status: active
Audience: single-user local-first operator
Purpose: describe one recommended PromptManager workflow that matches the current product center and current live UI.

## Why this path exists

PromptManager is strongest when it is used as a **local-first canonical home for prompt assets**.

This document describes one recommended operator path that keeps the product centered on:
- capture
- normalize
- retrieve
- inspect
- reuse
- refine

It does **not** treat analytics, chains, workbench, sharing, voice, or broad execution features as the default front door.

## Recommended operator path

### 1. Capture quickly with `Quick Capture`
Use **`Quick Capture`** from the main toolbar when you have raw prompt text or an LLM query that should not stay scattered in chat, notes, or scratch files.

Expected result:
- the raw text is saved as a **draft prompt**
- the prompt opens in the normal catalog/detail flow
- the asset is now inside the canonical prompt catalog rather than outside it

### 2. Normalize it with `Promote Draft`
If the captured item is worth keeping, use **`Promote Draft`** from the detail view.

Expected result:
- the draft becomes a normal reusable prompt asset
- title, category, tags, source, and note can be cleaned up without leaving the existing detail flow
- similar or duplicate cues may appear, but the operator stays in control

### 3. Reopen it later with `Recent` or search
When returning to the catalog, use:
- **`Recent`** for the fastest reopen path, or
- search when you already know what you are looking for

Expected result:
- the operator can get back to a recently touched prompt without rebuilding context from scratch

### 4. Inspect the asset in the detail view
Use the detail view to confirm:
- what the prompt is for
- where it came from
- whether it is still a draft or already reusable
- whether lineage or bounded comparison cues matter

Expected result:
- the operator can judge prompt fit without digging through raw metadata first

### 5. Reuse it with `Copy Prompt` or `Open in Workspace`
When the asset is ready to use, prefer the bounded quick-reuse actions:
- **`Copy Prompt`** when you want the stored prompt body directly
- **`Open in Workspace`** when lightweight validation or further reuse needs the workspace, without auto-running the prompt

Expected result:
- reuse stays fast
- validation remains optional
- the catalog stays the center of the product story

## What this path intentionally does not make primary

These surfaces can still be useful, but they are **not** the default front door for PromptManager:
- analytics
- chain workflows
- workbench-first authoring
- sharing flows
- voice/TTS
- web-enriched execution as a starting point

## Rule of thumb

If you are unsure where to start, use this sequence:

**`Quick Capture` → `Promote Draft` → `Recent` / search → inspect → `Copy Prompt` or `Open in Workspace`**

That is the canonical usage path for PromptManager v1.
