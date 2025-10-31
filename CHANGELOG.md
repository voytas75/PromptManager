# Changelog

All notable changes to **Prompt Manager** will be documented in this file.

## [0.5.0] - 2025-11-05

- Import the packaged prompt catalogue (or a user-supplied JSON directory) into SQLite and ChromaDB on startup.
- Add category, tag, and minimum-quality filters to the GUI alongside seeded prompt metadata.
- Expose configurable `catalog_path` setting and environment variable override.
- Package the built-in catalogue for distribution and provide catalogue import helpers with unit tests.

## [0.4.1] - 2025-11-05

- Promote GUI launch to the default `python -m main` behaviour with `--no-gui` opt-out.
- Add graceful shutdown hooks for the `PromptManager` background worker and backends.
- Split GUI requirements into an optional extra (`pip install .[gui]`) and dedicated requirements file.
- Suppress noisy Chroma PostHog telemetry while keeping collection initialisation intact.
- Improve packaging metadata and configure setuptools package discovery for `pip install .`.
- Extend tests to cover GUI dependency fallbacks, offscreen detection, and manager shutdown.
