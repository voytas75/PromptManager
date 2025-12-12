"""Utility script to load and print Prompt Manager settings for diagnostics.

Updates:
  v0.1.1 - 2025-12-12 - Exit non-zero when settings validation fails.
  v0.1.0 - 2025-12-12 - Add validation helper for environment/config debugging.
"""

from __future__ import annotations

import traceback

from config.settings import SettingsError, load_settings


def main() -> int:
    """Load settings and report validation outcomes."""
    try:
        settings = load_settings()
    except SettingsError:
        traceback.print_exc()
        return 2
    print("Settings loaded successfully.")
    print(f"prompt_output_font_color={settings.prompt_output_font_color}")
    print(f"chat_font_color={settings.chat_font_color}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
