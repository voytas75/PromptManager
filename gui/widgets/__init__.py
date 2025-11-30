"""Reusable Qt widgets and layouts shared across Prompt Manager.

Updates:
  v0.1.0 - 2025-11-30 - Introduce widgets package for shared Qt components.
"""
from .flow_layout import FlowLayout
from .prompt_detail_widget import PromptDetailWidget

__all__ = ["FlowLayout", "PromptDetailWidget"]
