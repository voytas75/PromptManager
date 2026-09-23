# Linux / WSL GUI Startup Reliability v1

Status: completed locally; delivery approval pending
Owner: Wojtek / Prompt Manager Team
Date: 2026-09-22

## Decision

Make the Linux X11/WSL Qt startup failure actionable before Qt constructs `QApplication`, and remove the confirmed duplicate-layout warning from Settings.

## Evidence

On a second user account in the same WSL instance, Qt found PySide6's `xcb` platform plugin but could not load it because `libxcb-icccm.so.4` was unresolved. Qt then terminated before the normal Python launcher could report a useful recovery action. The generic cursor-library message was not the root cause in that case.

The Settings dialog also emitted `QLayout: Attempting to add QLayout ... SettingsDialog ... which already has a layout`; inspection found two top-level `QVBoxLayout(self)` instances.

## Scope

- Linux-only preflight of PySide6's bundled `libqxcb.so` using `ldd`.
- A controlled launcher error that names unresolved libraries and gives the Ubuntu/Debian runtime package command.
- No automatic package installation, `sudo`, environment mutation, provider call, persistence change, or CLI command.
- Skip the xcb check for explicit non-xcb platforms and Wayland-only/headless contexts.
- Remove the duplicate top-level SettingsDialog layout while preserving the existing widget hierarchy.
- Document the Linux/WSL runtime prerequisite in user and developer startup guidance.

## Verification

- Focused provider-free GUI/startup regressions: `14 passed`.
- Ruff check and format check on touched Python files: pass.
- Scoped Pyright excluding inherited `gui/settings_dialog.py` debt: `0 errors`.
- Full provider-free suite: `892 passed, 1 skipped`; core coverage `81.68%`.
- CI-scope Pyright (`main.py config models`), `uv lock --check`, and `git diff --check`: pass.
- Controlled preflight smoke returned the intended actionable `libxcb-icccm.so.4` guidance.
- Offscreen bootstrap smoke completed successfully.

## Non-goals

- Do not require or install `xdg-desktop-portal`; its absence is a non-blocking desktop-integration warning.
- Do not change Windows/macOS GUI behavior.
- Do not redesign Settings or add a dashboard/health panel.

## Rollback

Revert the preflight helper, launcher error export/handling, SettingsDialog layout consolidation, and associated docs/tests as one bounded slice.
