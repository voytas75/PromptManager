"""Helper utilities for bootstrapping the Prompt Manager main window.

Updates:
  v0.15.82 - 2025-12-07 - Register the Rentry share provider alongside ShareText and PrivateBin.
  v0.15.81 - 2025-12-01 - Extracted widget/controller bootstrap helpers from gui.main_window.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel, QWidget

from core.sharing import PrivateBinProvider, RentryProvider, ShareTextProvider

from .appearance_controller import AppearanceController
from .layout_controller import LayoutController
from .layout_state import WindowStateManager
from .prompt_list_coordinator import PromptListCoordinator
from .prompt_list_model import PromptListModel
from .runtime_settings_service import RuntimeSettingsService
from .share_controller import ShareController
from .share_workflow import ShareWorkflowCoordinator
from .usage_logger import IntentUsageLogger
from .widgets import PromptDetailWidget

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable

    from config import PromptManagerSettings
    from core import PromptManager
    from models.prompt_model import Prompt
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    PromptManagerSettings = _Any
    PromptManager = _Any
    Prompt = _Any


class _PromptSupplier(Protocol):
    def __call__(self) -> object | None:
        """Return the currently selected prompt or widget."""


@dataclass(slots=True)
class DetailWidgetCallbacks:
    """Signal handlers used while wiring the prompt detail widget."""

    delete_requested: Callable[[], None]
    edit_requested: Callable[[], None]
    version_history_requested: Callable[[Prompt | None], None]
    fork_requested: Callable[[], None]
    refresh_scenarios_requested: Callable[[PromptDetailWidget], None]
    share_requested: Callable[[], None]


@dataclass(slots=True)
class BootstrapResult:
    """Container describing the collaborators created during bootstrap."""

    model: PromptListModel
    detail_widget: PromptDetailWidget
    prompt_coordinator: PromptListCoordinator
    usage_logger: IntentUsageLogger
    share_controller: ShareController
    share_workflow: ShareWorkflowCoordinator
    layout_state: WindowStateManager
    layout_controller: LayoutController
    runtime_settings_service: RuntimeSettingsService
    runtime_settings: dict[str, object | None]
    appearance_controller: AppearanceController
    notification_indicator: QLabel


class MainWindowBootstrapper:
    """Create long-lived helpers required by :class:`gui.main_window.MainWindow`."""

    def __init__(
        self,
        *,
        parent: QWidget,
        manager: PromptManager,
        settings: PromptManagerSettings | None,
        detail_callbacks: DetailWidgetCallbacks,
        toast_callback: Callable[[str, int], None],
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
        share_result_button_supplier: _PromptSupplier,
        execution_controller_supplier: _PromptSupplier,
    ) -> None:
        """Store dependencies required to construct bootstrap collaborators."""
        self._parent = parent
        self._manager = manager
        self._settings = settings
        self._detail_callbacks = detail_callbacks
        self._toast_callback = toast_callback
        self._status_callback = status_callback
        self._error_callback = error_callback
        self._share_result_button_supplier = share_result_button_supplier
        self._execution_controller_supplier = execution_controller_supplier

    def bootstrap(self) -> BootstrapResult:
        """Instantiate widgets, controllers, and services shared across the GUI."""
        model = PromptListModel(parent=self._parent)
        detail_widget = PromptDetailWidget(self._parent)
        detail_widget.delete_requested.connect(self._detail_callbacks.delete_requested)  # type: ignore[arg-type]
        detail_widget.edit_requested.connect(self._detail_callbacks.edit_requested)  # type: ignore[arg-type]
        detail_widget.version_history_requested.connect(
            self._detail_callbacks.version_history_requested
        )  # type: ignore[arg-type]
        detail_widget.fork_requested.connect(self._detail_callbacks.fork_requested)  # type: ignore[arg-type]
        detail_widget.refresh_scenarios_requested.connect(  # type: ignore[arg-type]
            partial(self._detail_callbacks.refresh_scenarios_requested, detail_widget)
        )
        detail_widget.share_requested.connect(self._detail_callbacks.share_requested)  # type: ignore[arg-type]

        runtime_settings_service = RuntimeSettingsService(self._manager, self._settings)
        runtime_settings = runtime_settings_service.build_initial_runtime_settings()
        appearance_controller = AppearanceController(self._parent, runtime_settings)
        usage_logger = IntentUsageLogger()

        share_controller = ShareController(
            self._parent,
            toast_callback=self._toast_callback,
            status_callback=self._status_callback,
            error_callback=self._error_callback,
            usage_logger=usage_logger,
            preference_supplier=lambda: bool(runtime_settings.get("auto_open_share_links", True)),
        )
        share_workflow = ShareWorkflowCoordinator(
            share_controller,
            detail_widget=detail_widget,
            prompt_supplier=detail_widget.current_prompt,
            share_button_supplier=detail_widget.share_button,
            share_result_button_supplier=self._share_result_button_supplier,
            execution_controller_supplier=self._execution_controller_supplier,
            status_callback=self._status_callback,
            error_callback=self._error_callback,
        )
        share_workflow.register_provider(ShareTextProvider())
        share_workflow.register_provider(RentryProvider())
        if self._settings is not None:
            privatebin_provider = PrivateBinProvider(
                base_url=self._settings.privatebin_url,
                expiration=self._settings.privatebin_expiration,
                formatter=self._settings.privatebin_format,
                compression=self._settings.privatebin_compression,
                burn_after_reading=self._settings.privatebin_burn_after_reading,
                open_discussion=self._settings.privatebin_open_discussion,
            )
        else:
            privatebin_provider = PrivateBinProvider()
        share_workflow.register_provider(privatebin_provider)

        settings_obj = QSettings("PromptManager", "MainWindow")
        layout_state = WindowStateManager(settings_obj)
        layout_controller = LayoutController(layout_state)
        notification_indicator = QLabel("", self._parent)
        notification_indicator.setObjectName("notificationIndicator")
        notification_indicator.setStyleSheet("color: #2f80ed; font-weight: 500;")
        notification_indicator.setVisible(False)

        return BootstrapResult(
            model=model,
            detail_widget=detail_widget,
            prompt_coordinator=PromptListCoordinator(self._manager),
            usage_logger=usage_logger,
            share_controller=share_controller,
            share_workflow=share_workflow,
            layout_state=layout_state,
            layout_controller=layout_controller,
            runtime_settings_service=runtime_settings_service,
            runtime_settings=runtime_settings,
            appearance_controller=appearance_controller,
            notification_indicator=notification_indicator,
        )


__all__ = [
    "BootstrapResult",
    "DetailWidgetCallbacks",
    "MainWindowBootstrapper",
]
