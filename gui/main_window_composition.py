"""Composable helpers for wiring the Prompt Manager main window.

Updates:
  v0.15.82 - 2025-12-01 - Introduced composition helpers for gui.main_window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .main_window_bootstrapper import DetailWidgetCallbacks, MainWindowBootstrapper
from .prompt_generation_service import PromptGenerationService
from .prompt_search_controller import PromptSearchController

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from PySide6.QtWidgets import QWidget

    from config import PromptManagerSettings
    from core import PromptManager
    from models.prompt_model import Prompt

    from .controllers.execution_controller import ExecutionController
    from .main_window_bootstrapper import BootstrapResult
    from .prompt_list_presenter import PromptListPresenter
    from .widgets import PromptFilterPanel
else:  # pragma: no cover - runtime placeholders for type-only imports
    QWidget = object  # type: ignore[assignment]
    PromptManagerSettings = object  # type: ignore[assignment]
    BootstrapResult = object  # type: ignore[assignment]
    Prompt = object  # type: ignore[assignment]
    PromptListPresenter = object  # type: ignore[assignment]
    PromptFilterPanel = object  # type: ignore[assignment]
    ExecutionController = object  # type: ignore[assignment]


@dataclass(slots=True)
class FilterPreferences:
    """Persisted filters restored during main window initialisation."""

    category_slug: str | None
    tag: str | None
    min_quality: int | None
    sort_value: str | None


@dataclass(slots=True)
class PromptGenerationHooks:
    """Callables required to build prompt editor workflows."""

    execute_prompt_as_context: Callable[[Prompt, QWidget | None, str | None], None]
    load_prompts: Callable[[str], None]
    current_search_text: Callable[[], str]
    select_prompt: Callable[[UUID], None]
    delete_prompt: Callable[[Prompt], None]
    status_callback: Callable[[str, int], None]
    error_callback: Callable[[str, str], None]
    current_prompt_supplier: Callable[[], Prompt | None]
    open_version_history_dialog: Callable[[Prompt | None], None]


@dataclass(slots=True)
class PromptSearchHooks:
    """Lazy suppliers for prompt search controller wiring."""

    presenter_supplier: Callable[[], PromptListPresenter | None]
    filter_panel_supplier: Callable[[], PromptFilterPanel | None]
    load_prompts: Callable[[str], None]
    current_search_text: Callable[[], str]
    select_prompt: Callable[[UUID], None]


@dataclass(slots=True)
class MainWindowComposition:
    """Bundle describing collaborators needed by :class:`gui.main_window.MainWindow`."""

    bootstrap: BootstrapResult
    prompt_generation_service: PromptGenerationService
    prompt_search_controller: PromptSearchController
    filter_preferences: FilterPreferences


def build_main_window_composition(
    *,
    parent: QWidget,
    manager: PromptManager,
    settings: PromptManagerSettings | None,
    detail_callbacks: DetailWidgetCallbacks,
    toast_callback: Callable[[str, int], None],
    status_callback: Callable[[str, int], None],
    error_callback: Callable[[str, str], None],
    prompt_generation_hooks: PromptGenerationHooks,
    prompt_search_hooks: PromptSearchHooks,
    share_result_button_supplier: Callable[[], object | None],
    execution_controller_supplier: Callable[[], ExecutionController | None],
) -> MainWindowComposition:
    """Create bootstrap collaborators and supporting controllers."""
    bootstrapper = MainWindowBootstrapper(
        parent=parent,
        manager=manager,
        settings=settings,
        detail_callbacks=detail_callbacks,
        toast_callback=toast_callback,
        status_callback=status_callback,
        error_callback=error_callback,
        share_result_button_supplier=share_result_button_supplier,
        execution_controller_supplier=execution_controller_supplier,
    )
    bootstrap = bootstrapper.bootstrap()

    prompt_generation_service = PromptGenerationService(
        manager=manager,
        execute_context_handler=prompt_generation_hooks.execute_prompt_as_context,
        load_prompts=prompt_generation_hooks.load_prompts,
        current_search_text=prompt_generation_hooks.current_search_text,
        select_prompt=prompt_generation_hooks.select_prompt,
        delete_prompt=prompt_generation_hooks.delete_prompt,
        status_callback=prompt_generation_hooks.status_callback,
        error_callback=prompt_generation_hooks.error_callback,
        current_prompt_supplier=prompt_generation_hooks.current_prompt_supplier,
        open_version_history_dialog=prompt_generation_hooks.open_version_history_dialog,
    )

    prompt_search_controller = PromptSearchController(
        parent=parent,
        manager=manager,
        presenter_supplier=prompt_search_hooks.presenter_supplier,
        filter_panel_supplier=prompt_search_hooks.filter_panel_supplier,
        layout_controller=bootstrap.layout_controller,
        load_prompts=prompt_search_hooks.load_prompts,
        current_search_text=prompt_search_hooks.current_search_text,
        select_prompt=prompt_search_hooks.select_prompt,
    )

    filter_state = bootstrap.layout_state.load_filter_preferences()
    filter_preferences = FilterPreferences(
        category_slug=filter_state.category_slug,
        tag=filter_state.tag,
        min_quality=filter_state.min_quality,
        sort_value=filter_state.sort_value,
    )

    return MainWindowComposition(
        bootstrap=bootstrap,
        prompt_generation_service=prompt_generation_service,
        prompt_search_controller=prompt_search_controller,
        filter_preferences=filter_preferences,
    )


__all__ = [
    "FilterPreferences",
    "MainWindowComposition",
    "PromptGenerationHooks",
    "PromptSearchHooks",
    "build_main_window_composition",
]
