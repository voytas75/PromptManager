"""Workspace insight controller for language detection and prompt suggestions.

Updates:
  v0.15.82 - 2025-12-01 - Extract workspace insight logic from gui.main_window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core import IntentLabel, PromptManager, PromptManagerError

from .language_tools import DetectedLanguage, detect_language

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Sequence

    from PySide6.QtWidgets import QLabel, QPlainTextEdit

    from models.prompt_model import Prompt

    from .code_highlighter import CodeHighlighter
    from .prompt_list_presenter import PromptListPresenter
    from .quick_action_controller import QuickActionController
    from .usage_logger import IntentUsageLogger


class WorkspaceInsightController:
    """Handle workspace language detection, intent hints, and suggestions."""

    def __init__(
        self,
        *,
        manager: PromptManager,
        query_input: QPlainTextEdit,
        intent_hint_label: QLabel,
        language_label: QLabel,
        highlighter: CodeHighlighter,
        presenter_supplier: Callable[[], PromptListPresenter | None],
        current_prompts_supplier: Callable[[], Sequence[Prompt]],
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
        usage_logger: IntentUsageLogger,
        quick_action_controller_supplier: Callable[[], QuickActionController | None],
        current_search_text: Callable[[], str],
    ) -> None:
        """Capture widgets, controllers, and callbacks required for insights."""
        self._manager = manager
        self._query_input = query_input
        self._intent_hint_label = intent_hint_label
        self._language_label = language_label
        self._highlighter = highlighter
        self._presenter_supplier = presenter_supplier
        self._current_prompts_supplier = current_prompts_supplier
        self._status_callback = status_callback
        self._error_callback = error_callback
        self._usage_logger = usage_logger
        self._quick_action_controller_supplier = quick_action_controller_supplier
        self._current_search_text = current_search_text
        self._detected_language: DetectedLanguage = detect_language("")

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def initialise_language(self) -> None:
        """Seed detection state based on the current workspace text."""
        self._update_detected_language(self._query_input.toPlainText(), force=True)

    def handle_query_text_changed(self) -> None:
        """Refresh detection and clear quick-action workspace seeds when needed."""
        text = self._query_input.toPlainText()
        self._update_detected_language(text)
        controller = self._quick_action_controller_supplier()
        if controller is None:
            return
        if controller.is_workspace_signal_suppressed():
            return
        controller.clear_workspace_seed()

    # ------------------------------------------------------------------
    # Intent + suggestion helpers
    # ------------------------------------------------------------------
    def detect_intent(self) -> None:
        """Run the classifier over the workspace text and update hints."""
        text = self._query_input.toPlainText().strip()
        if not text:
            self._status_callback("Paste some text or code to analyse.", 4000)
            return
        classifier = self._manager.intent_classifier
        if classifier is None:
            self._error_callback(
                "Intent detection unavailable",
                "No intent classifier is configured.",
            )
            return
        prediction = classifier.classify(text)
        current_prompts = list(self._current_prompts_supplier())
        presenter = self._presenter_supplier()
        if presenter is not None:
            presenter.set_intent_suggestions(
                PromptManager.IntentSuggestions(  # type: ignore[attr-defined]
                    prediction=prediction,
                    prompts=list(current_prompts),
                    fallback_used=False,
                )
            )
        self.update_intent_hint(current_prompts)
        label = prediction.label.value.replace("_", " ").title()
        self._status_callback(
            f"Detected intent: {label} ({int(prediction.confidence * 100)}%)",
            5000,
        )
        self._usage_logger.log_detect(prediction=prediction, query_text=text)

    def suggest_prompt(self) -> None:
        """Produce prompt suggestions using the query input or toolbar search."""
        query = self._query_input.toPlainText().strip() or self._current_search_text().strip()
        if not query:
            self._status_callback("Provide text or use search to fetch suggestions.", 4000)
            return
        try:
            suggestions = self._manager.suggest_prompts(query, limit=20)
        except PromptManagerError as exc:
            self._error_callback("Unable to suggest prompts", str(exc))
            return
        self._usage_logger.log_suggest(
            prediction=suggestions.prediction,
            query_text=query,
            prompts=suggestions.prompts,
            fallback_used=suggestions.fallback_used,
        )
        presenter = self._presenter_supplier()
        if presenter is not None:
            presenter.apply_suggestions(suggestions)
        top_name = suggestions.prompts[0].name if suggestions.prompts else None
        if top_name:
            self._status_callback(f"Top suggestion: {top_name}", 5000)

    def update_intent_hint(self, prompts: Sequence[Prompt]) -> None:
        """Update the intent hint label using suggestion metadata."""
        presenter = self._presenter_supplier()
        suggestions = presenter.suggestions if presenter is not None else None
        if suggestions is None:
            self._intent_hint_label.clear()
            self._intent_hint_label.setVisible(False)
            return

        prompt_list = list(prompts)
        prediction = suggestions.prediction
        label_text = prediction.label.value.replace("_", " ").title()
        confidence_pct = int(round(prediction.confidence * 100))

        summary_parts: list[str] = []
        if prediction.label is not IntentLabel.GENERAL or prediction.category_hints:
            summary_parts.append(f"Detected intent: {label_text} ({confidence_pct}%).")
        if prediction.category_hints:
            summary_parts.append(f"Focus: {', '.join(prediction.category_hints)}")
        if prediction.language_hints:
            summary_parts.append(f"Lang: {', '.join(prediction.language_hints)}")

        top_names = [prompt.name for prompt in prompt_list[:3]]
        if top_names:
            summary_parts.append(f"Top matches: {', '.join(top_names)}")
        if suggestions.fallback_used:
            summary_parts.append("Fallback ranking applied")

        if summary_parts:
            message = " | ".join(summary_parts)
        elif top_names:
            message = "Top matches: " + ", ".join(top_names)
        else:
            self._intent_hint_label.clear()
            self._intent_hint_label.setVisible(False)
            return

        self._intent_hint_label.setText(message)
        self._intent_hint_label.setVisible(True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_detected_language(self, text: str, *, force: bool = False) -> None:
        detection = detect_language(text)
        if not force and detection.code == self._detected_language.code:
            return
        self._detected_language = detection
        if detection.confidence:
            percentage = int(round(detection.confidence * 100))
            self._language_label.setText(f"Language: {detection.name} ({percentage}%)")
        else:
            self._language_label.setText(f"Language: {detection.name}")
        self._highlighter.set_language(detection.code)
        self._status_callback(f"Workspace language: {detection.name}", 3000)


__all__ = ["WorkspaceInsightController"]
