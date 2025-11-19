"""Data models for Prompt Manager.

Updates: v0.2.0 - 2025-12-05 - Export ResponseStyle dataclass.
Updates: v0.1.0 - 2025-10-30 - Export Prompt dataclass.
"""

from .prompt_model import Prompt
from .response_style import ResponseStyle

__all__ = ["Prompt", "ResponseStyle"]
