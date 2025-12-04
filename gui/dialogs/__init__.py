"""Dialog widgets used by the Prompt Manager GUI.

Updates:
  v0.12.1 - 2025-12-04 - Export prompt chain manager dialog for GUI workflows.
  v0.12.0 - 2025-12-03 - Split monolithic module into package modules and re-exported APIs.
  v0.11.14 - 2025-11-29 - Shorten update summaries and wrap maintenance tooltips.
  v0.11.13 - 2025-11-28 - Keep maintenance button pinned below scroll surface.
  v0.11.12 - 2025-11-28 - Add backup snapshot action to the maintenance dialog.
  v0.11.11 - 2025-11-28 - Make maintenance dialog scrollable with shorter default height.
  v0.11.10 - 2025-11-28 - Add prompt body diff tab for comparing selected versions.
  v0.11.9 - 2025-11-28 - Add category health analytics table to maintenance dialog.
  v0.11.8 - 2025-11-27 - Show toast confirmations for dialog copy actions.
  v0.11.7 - 2025-11-27 - Add copy prompt body control to version history dialog.
  v0.11.6 - 2025-11-27 - Add prompt part selector to response style dialog and rename copy action.
"""

from __future__ import annotations

from .base import (
    CollapsibleTextSection,
    fallback_generate_description,
    fallback_generate_scenarios,
    fallback_suggest_prompt_name,
    strip_scenarios_metadata,
)
from .catalog import CatalogPreviewDialog
from .categories import CategoryEditorDialog, CategoryManagerDialog
from .execution import ResponseStyleDialog, SaveResultDialog
from .history import PromptVersionHistoryDialog
from .maintenance import PromptMaintenanceDialog
from .notes import PromptNoteDialog
from .previews import InfoDialog, MarkdownPreviewDialog
from .prompt_chains import PromptChainManagerDialog
from .prompt_editor import PromptDialog, PromptRefinedDialog

_strip_scenarios_metadata = strip_scenarios_metadata

__all__ = [
    "CatalogPreviewDialog",
    "CategoryEditorDialog",
    "CategoryManagerDialog",
    "CollapsibleTextSection",
    "InfoDialog",
    "MarkdownPreviewDialog",
    "PromptDialog",
    "PromptMaintenanceDialog",
    "PromptNoteDialog",
    "PromptChainManagerDialog",
    "PromptRefinedDialog",
    "PromptVersionHistoryDialog",
    "ResponseStyleDialog",
    "SaveResultDialog",
    "_strip_scenarios_metadata",
    "fallback_generate_description",
    "fallback_generate_scenarios",
    "fallback_suggest_prompt_name",
]
