# Prompt Manager

Prompt Manager is a desktop-focused application for cataloguing, searching, and executing AI prompts. The project follows the architecture captured in `PromptManagerBlueprint.txt` and targets PySide6 for the GUI while relying on SQLite for canonical storage, ChromaDB for semantic retrieval, and Redis for responsive caching.

## Current Status

- Core data model defined in `models/prompt_model.py` with an extensible schema (`ext1`–`ext5`) and serialization helpers.
- `PromptRepository` (`core/repository.py`) persists prompts to SQLite and exposes CRUD operations.
- `PromptManager` service (`core/prompt_manager.py`) coordinates SQLite persistence, Redis-backed caching, and ChromaDB semantic search.
- Project scaffolding prepared for config, GUI, data, and test layers.

## Getting Started

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure SQLite and ChromaDB have filesystem access at the configured paths (defaults applied in code).

3. Provide a Redis instance (local Docker or managed) and pass the client into `PromptManager` to enable caching. Without Redis the manager still operates with SQLite as the authoritative store and ChromaDB for semantic lookups.

4. Run the automated tests to validate storage integration:

   ```bash
   pytest
   ```

Further modules (GUI, session history, execution pipeline) will be introduced in subsequent milestones in line with the blueprint.
