"""Coordinate prompt selection, lineage, and template preview updates.

Updates:
  v0.15.88 - 2026-04-27 - Align limited-evidence next actions to action-oriented inspect wording.
  v0.15.87 - 2026-04-26 - Add bounded validation freshness cues to inspect run summaries.
  v0.15.86 - 2026-04-25 - Centralize bounded decision -> next-action mapping for inspect cues.
  v0.15.85 - 2026-04-10 - Add bounded candidate-vs-baseline comparison cues to run summaries.
  v0.15.84 - 2026-04-10 - Add bounded changed-from-parent lineage cue for forked prompts.
  v0.15.83 - 2026-04-10 - Resolve parent lineage summaries to human-readable prompt names.
"""


from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from core import PromptManager, PromptManagerError, PromptVersionError

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from PySide6.QtWidgets import QListView

    from models.prompt_model import Prompt

    from .controllers.execution_controller import ExecutionController
    from .prompt_list_model import PromptListModel
    from .template_preview_controller import TemplatePreviewController
    from .widgets import PromptDetailWidget
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    QListView = _Any
    Prompt = _Any
    ExecutionController = _Any
    PromptListModel = _Any
    TemplatePreviewController = _Any
    PromptDetailWidget = _Any


class WorkspaceHistoryController:
    """Encapsulate prompt selection, lineage, and template preview tasks."""

    def __init__(
        self,
        *,
        manager: PromptManager,
        model: PromptListModel,
        detail_widget: PromptDetailWidget,
        list_view: QListView,
        current_prompt_supplier: Callable[[], Prompt | None],
        template_detail_widget_supplier: Callable[[], PromptDetailWidget | None],
        template_preview_controller_supplier: Callable[[], TemplatePreviewController | None],
        execution_controller_supplier: Callable[[], ExecutionController | None],
    ) -> None:
        """Store collaborators required to synchronize selection + lineage state."""
        self._manager = manager
        self._model = model
        self._detail_widget = detail_widget
        self._list_view = list_view
        self._current_prompt_supplier = current_prompt_supplier
        self._template_detail_widget_supplier = template_detail_widget_supplier
        self._template_preview_controller_supplier = template_preview_controller_supplier
        self._execution_controller_supplier = execution_controller_supplier

    def handle_selection_changed(self) -> None:
        """Update detail + template panels when the current prompt changes."""
        prompt = self._current_prompt_supplier()
        if prompt is None:
            self._detail_widget.clear()
            execution_controller = self._execution_controller_supplier()
            if execution_controller is not None:
                execution_controller.handle_prompt_selection_change(None)
            self._update_template_preview(None)
            template_detail = self._template_detail_widget_supplier()
            if template_detail is not None:
                template_detail.clear()
            return

        execution_controller = self._execution_controller_supplier()
        if execution_controller is not None:
            execution_controller.handle_prompt_selection_change(prompt.id)

        self._detail_widget.display_prompt(prompt)
        template_detail = self._template_detail_widget_supplier()
        if template_detail is not None:
            template_detail.display_prompt(prompt)
        self._update_prompt_lineage_summary(prompt)
        self._update_prompt_decision_summary(prompt)
        self._update_prompt_next_action_summary(prompt)
        self._update_prompt_run_summary(prompt)
        self._update_template_preview(prompt)

    def select_prompt(self, prompt_id: UUID) -> None:
        """Highlight *prompt_id* in the list view when present."""
        for row, prompt in enumerate(self._model.prompts()):
            if prompt.id == prompt_id:
                index = self._model.index(row, 0)
                self._list_view.setCurrentIndex(index)
                break

    def _update_prompt_lineage_summary(self, prompt: Prompt) -> None:
        summary_parts: list[str] = []
        try:
            parent_link = self._manager.get_prompt_parent_fork(prompt.id)
        except PromptVersionError:
            parent_link = None
        if parent_link is not None:
            parent_label = self._resolve_prompt_name(parent_link.source_prompt_id)
            summary_parts.append(f"Forked from {parent_label}")
            difference_summary = self._build_parent_difference_summary(
                prompt=prompt,
                parent_prompt_id=parent_link.source_prompt_id,
            )
            if difference_summary is not None:
                summary_parts.append(difference_summary)

        try:
            children = self._manager.list_prompt_forks(prompt.id)
        except PromptVersionError:
            children = []
        if children:
            child_label = "fork" if len(children) == 1 else "forks"
            summary_parts.append(f"{len(children)} {child_label}")
        summary_text = " | ".join(summary_parts) if summary_parts else "No lineage data yet."
        self._detail_widget.update_lineage_summary(summary_text)
        template_detail = self._template_detail_widget_supplier()
        if template_detail is not None:
            template_detail.update_lineage_summary(summary_text)

    def _build_parent_difference_summary(
        self,
        *,
        prompt: Prompt,
        parent_prompt_id: UUID,
    ) -> str | None:
        try:
            parent_prompt = self._manager.get_prompt(parent_prompt_id)
        except PromptManagerError:
            return None

        changed_fields = self._changed_fields_against_parent(
            prompt=prompt,
            parent_prompt=parent_prompt,
        )
        if not changed_fields:
            return None
        return f"Changed from parent: {', '.join(changed_fields)}"

    def _update_prompt_decision_summary(self, prompt: Prompt) -> None:
        """Render one bounded inspect recommendation using existing lineage state."""
        decision_text = self._build_decision_summary(prompt)
        self._detail_widget.update_decision_summary(decision_text)
        template_detail = self._template_detail_widget_supplier()
        if template_detail is not None:
            template_detail.update_decision_summary(decision_text)
        provenance_text = self._build_decision_provenance_summary(prompt)
        self._detail_widget.update_decision_provenance_summary(provenance_text)
        if template_detail is not None:
            template_detail.update_decision_provenance_summary(provenance_text)

    def _update_prompt_next_action_summary(self, prompt: Prompt) -> None:
        """Render one bounded operator-facing next action from existing decision evidence."""
        next_action_text = self._build_next_action_summary(prompt)
        self._detail_widget.update_next_action_summary(next_action_text)
        template_detail = self._template_detail_widget_supplier()
        if template_detail is not None:
            template_detail.update_next_action_summary(next_action_text)

    def _update_prompt_run_summary(self, prompt: Prompt) -> None:
        """Render one bounded last-run evidence cue using existing execution history."""
        run_summary = self._build_run_summary(prompt)
        self._detail_widget.update_run_summary(run_summary)
        template_detail = self._template_detail_widget_supplier()
        if template_detail is not None:
            template_detail.update_run_summary(run_summary)

    def _build_run_summary(self, prompt: Prompt) -> str | None:
        entries = self._list_execution_history(prompt, limit=5)
        if not entries:
            return None
        latest = entries[0]
        metadata = getattr(latest, "metadata", None) or {}
        context = metadata.get("context", {}) if isinstance(metadata, dict) else {}
        execution = context.get("execution", {}) if isinstance(context, dict) else {}
        run = context.get("run", {}) if isinstance(context, dict) else {}

        status_value = getattr(getattr(latest, "status", None), "value", None) or "unknown"
        model = execution.get("model") if isinstance(execution, dict) else None
        prompt_version = run.get("prompt_version") if isinstance(run, dict) else None
        conversation_messages = run.get("conversation_messages") if isinstance(run, dict) else None
        duration_ms = getattr(latest, "duration_ms", None)

        parts = [f"Last run: {status_value}"]
        if model:
            parts[0] = f"{parts[0]} via {model}"
        if prompt_version is not None:
            parts.append(f"v{prompt_version}")
        if conversation_messages is not None:
            message_label = "message" if int(conversation_messages) == 1 else "messages"
            parts.append(f"{conversation_messages} {message_label}")
        if duration_ms is not None:
            parts.append(f"{duration_ms} ms")

        freshness_summary = self._build_validation_freshness_summary(latest)
        if freshness_summary is not None:
            parts.append(freshness_summary)

        comparison_readiness_summary = self._build_comparison_readiness_summary(entries)
        if comparison_readiness_summary is not None:
            parts.append(comparison_readiness_summary)

        comparison_summary = self._build_run_comparison_summary(entries)
        if comparison_summary is not None:
            parts.append(comparison_summary)
        return " · ".join(parts)

    def _build_comparison_readiness_summary(self, entries: list[object]) -> str | None:
        if len(entries) < 2:
            return None
        latest = entries[0]
        baseline = entries[1]
        latest_prompt_version = self._extract_prompt_version(latest)
        baseline_prompt_version = self._extract_prompt_version(baseline)
        if latest_prompt_version is None or baseline_prompt_version is None:
            return None
        if latest_prompt_version == baseline_prompt_version:
            return "Comparison readiness: no baseline yet"

        latest_rating = getattr(latest, "rating", None)
        baseline_rating = getattr(baseline, "rating", None)
        latest_duration = getattr(latest, "duration_ms", None)
        baseline_duration = getattr(baseline, "duration_ms", None)
        if (
            latest_rating is None
            or baseline_rating is None
            or latest_duration is None
            or baseline_duration is None
        ):
            return "Comparison readiness: limited"
        return None

    def _build_run_comparison_summary(self, entries: list[object]) -> str | None:
        if len(entries) < 2:
            return None
        latest = entries[0]
        baseline = entries[1]
        latest_prompt_version = self._extract_prompt_version(latest)
        baseline_prompt_version = self._extract_prompt_version(baseline)
        if latest_prompt_version is None or baseline_prompt_version is None:
            return None
        if latest_prompt_version == baseline_prompt_version:
            return None
        latest_rating = getattr(latest, "rating", None)
        baseline_rating = getattr(baseline, "rating", None)
        latest_duration = getattr(latest, "duration_ms", None)
        baseline_duration = getattr(baseline, "duration_ms", None)
        if latest_rating is None or baseline_rating is None:
            return None
        if latest_duration is None or baseline_duration is None:
            return None

        outcome = "improved" if float(latest_rating) > float(baseline_rating) else "regressed"
        if float(latest_rating) == float(baseline_rating):
            outcome = "matched"
        return (
            "Candidate vs baseline: "
            f"{outcome} (rating {latest_rating:.1f} vs {baseline_rating:.1f}; "
            f"{latest_duration} ms vs {baseline_duration} ms)"
        )

    @staticmethod
    def _build_validation_freshness_summary(entry: object) -> str | None:
        """Return one compact freshness cue from the latest execution timestamp when available."""
        executed_at = getattr(entry, "executed_at", None)
        if not isinstance(executed_at, datetime):
            return None
        if executed_at.tzinfo is None:
            return None
        freshness_window_seconds = 3 * 24 * 60 * 60
        age_seconds = (datetime.now(UTC) - executed_at.astimezone(UTC)).total_seconds()
        if age_seconds < 0:
            return None
        freshness = "recent" if age_seconds <= freshness_window_seconds else "stale"
        return f"Validation freshness: {freshness}"

    def _list_execution_history(self, prompt: Prompt, *, limit: int = 1) -> list[object]:
        list_history = getattr(self._manager, "list_execution_history", None)
        if not callable(list_history):
            return []
        try:
            entries = list_history(prompt.id, limit=limit)
        except PromptManagerError:
            return []
        if not isinstance(entries, list):
            return []
        return entries

    def _build_decision_summary(self, prompt: Prompt) -> str:
        run_recommendation = self._build_run_recommendation_summary(prompt)
        if run_recommendation is not None:
            return run_recommendation
        try:
            parent_link = self._manager.get_prompt_parent_fork(prompt.id)
        except PromptVersionError:
            parent_link = None

        if parent_link is not None:
            try:
                parent_prompt = self._manager.get_prompt(parent_link.source_prompt_id)
            except PromptManagerError:
                return "Fork before editing"
            changed_fields = self._changed_fields_against_parent(
                prompt=prompt,
                parent_prompt=parent_prompt,
            )
            if changed_fields:
                return "Refine before reuse"
            return "Fork before editing"
        return "Reuse as-is"

    def _build_decision_provenance_summary(self, prompt: Prompt) -> str | None:
        """Describe the bounded evidence source behind the current decision when it helps."""
        if self._build_run_recommendation_summary(prompt) is not None:
            return "Decision based on latest 2 comparable runs"
        if self._build_missing_evidence_reason(prompt) is not None:
            return "Decision based on limited run evidence"
        try:
            parent_link = self._manager.get_prompt_parent_fork(prompt.id)
        except PromptVersionError:
            parent_link = None
        if parent_link is not None:
            return "Decision based on fork lineage only"
        return None

    def _build_next_action_summary(self, prompt: Prompt) -> str:
        """Map bounded decision and evidence gaps to one compact operator-facing action."""
        stale_validation_action = self._build_stale_validation_next_action(prompt)
        if stale_validation_action is not None:
            return stale_validation_action
        missing_evidence_action = self._build_missing_evidence_next_action(prompt)
        if missing_evidence_action is not None:
            return missing_evidence_action
        decision_text = self._build_decision_summary(prompt)
        return self._map_decision_to_next_action(decision_text)

    @staticmethod
    def _map_decision_to_next_action(decision_text: str) -> str:
        """Translate bounded decision wording into one compact operator-facing action."""
        if decision_text == "Compare improved run":
            return "Validate improved run before reuse"
        if decision_text == "Compare regressed run":
            return "Validate baseline before reuse"
        if decision_text == "Keep baseline":
            return "Prefer baseline before reuse"
        if decision_text == "Matched baseline":
            return "Validate before reuse"
        if decision_text == "Refine before reuse":
            return "Refine before reuse"
        if decision_text == "Fork before editing":
            return "Fork before editing"
        return "Reuse as-is"

    def _build_stale_validation_next_action(self, prompt: Prompt) -> str | None:
        """Prefer a validation-first next step when only stale single-run evidence exists."""
        entries = self._list_execution_history(prompt, limit=2)
        if len(entries) != 1:
            return None
        if self._build_validation_freshness_summary(entries[0]) != "Validation freshness: stale":
            return None
        if self._build_decision_summary(prompt) != "Reuse as-is":
            return None
        return "Validate before reuse"

    def _build_missing_evidence_next_action(self, prompt: Prompt) -> str | None:
        """Translate limited-evidence inspect gaps into one compact operator-facing action."""
        missing_evidence_reason = self._build_missing_evidence_reason(prompt)
        if missing_evidence_reason == "Evidence: only one run available":
            return "Validate before reuse"
        if missing_evidence_reason == "Evidence: no comparable baseline yet":
            return "Run another version before comparing"
        if missing_evidence_reason == "Evidence: missing rating for comparison":
            return "Add ratings before comparing"
        if missing_evidence_reason == "Evidence: missing duration for comparison":
            return "Run again before comparing"
        return None

    def _build_run_recommendation_summary(self, prompt: Prompt) -> str | None:
        entries = self._list_execution_history(prompt, limit=2)
        if len(entries) < 2:
            return None
        latest = entries[0]
        baseline = entries[1]
        latest_prompt_version = self._extract_prompt_version(latest)
        baseline_prompt_version = self._extract_prompt_version(baseline)
        if latest_prompt_version is None or baseline_prompt_version is None:
            return None
        if latest_prompt_version == baseline_prompt_version:
            return None
        latest_rating = getattr(latest, "rating", None)
        baseline_rating = getattr(baseline, "rating", None)
        latest_duration = getattr(latest, "duration_ms", None)
        baseline_duration = getattr(baseline, "duration_ms", None)
        if latest_rating is None or baseline_rating is None:
            return None
        if latest_duration is None or baseline_duration is None:
            return None
        latest_rating_value = float(latest_rating)
        baseline_rating_value = float(baseline_rating)
        latest_duration_value = int(latest_duration)
        baseline_duration_value = int(baseline_duration)
        if (
            latest_rating_value == baseline_rating_value
            and latest_duration_value == baseline_duration_value
        ):
            return "Matched baseline"
        if latest_rating_value > baseline_rating_value:
            return "Compare improved run"
        if (
            latest_rating_value <= baseline_rating_value - 2.0
            and latest_duration_value >= baseline_duration_value + 100
        ):
            return "Keep baseline"
        return "Compare regressed run"

    def _build_missing_evidence_reason(self, prompt: Prompt) -> str | None:
        """Explain why bounded reuse evidence is too weak when comparison cues cannot be trusted."""
        entries = self._list_execution_history(prompt, limit=2)
        if not entries:
            return None
        if len(entries) < 2:
            return "Evidence: only one run available"
        latest = entries[0]
        baseline = entries[1]
        latest_prompt_version = self._extract_prompt_version(latest)
        baseline_prompt_version = self._extract_prompt_version(baseline)
        if latest_prompt_version is None or baseline_prompt_version is None:
            return "Evidence: no comparable baseline yet"
        if latest_prompt_version == baseline_prompt_version:
            return "Evidence: no comparable baseline yet"
        latest_rating = getattr(latest, "rating", None)
        baseline_rating = getattr(baseline, "rating", None)
        if latest_rating is None or baseline_rating is None:
            return "Evidence: missing rating for comparison"
        latest_duration = getattr(latest, "duration_ms", None)
        baseline_duration = getattr(baseline, "duration_ms", None)
        if latest_duration is None or baseline_duration is None:
            return "Evidence: missing duration for comparison"
        return None

    @staticmethod
    def _runs_form_comparable_pair(latest: object, baseline: object) -> bool:
        """Return whether two runs are meaningfully comparable as baseline/candidate evidence."""
        latest_duration = getattr(latest, "duration_ms", None)
        baseline_duration = getattr(baseline, "duration_ms", None)
        latest_rating = getattr(latest, "rating", None)
        baseline_rating = getattr(baseline, "rating", None)
        if latest_duration is None or baseline_duration is None:
            return False
        if latest_rating is None or baseline_rating is None:
            return False
        return int(latest_duration) != int(baseline_duration) or float(latest_rating) != float(
            baseline_rating
        )

    @staticmethod
    def _extract_prompt_version(entry: object) -> int | None:
        """Return the recorded prompt version from execution metadata when available."""
        metadata = getattr(entry, "metadata", None) or {}
        if not isinstance(metadata, dict):
            return None
        context = metadata.get("context", {})
        if not isinstance(context, dict):
            return None
        run = context.get("run", {})
        if not isinstance(run, dict):
            return None
        prompt_version = run.get("prompt_version")
        if prompt_version is None:
            return None
        try:
            return int(prompt_version)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _changed_fields_against_parent(*, prompt: Prompt, parent_prompt: Prompt) -> list[str]:
        changed_fields: list[str] = []
        if (prompt.context or "") != (parent_prompt.context or ""):
            changed_fields.append("body")
        if prompt.description != parent_prompt.description:
            changed_fields.append("description")
        if prompt.tags != parent_prompt.tags:
            changed_fields.append("tags")
        if prompt.source != parent_prompt.source:
            changed_fields.append("source")
        return changed_fields

    def _resolve_prompt_name(self, prompt_id: UUID) -> str:
        """Return a human-readable prompt label for lineage summaries."""
        try:
            parent_prompt = self._manager.get_prompt(prompt_id)
        except PromptManagerError:
            return str(prompt_id)
        name = parent_prompt.name.strip()
        return name or str(prompt_id)

    def _update_template_preview(self, prompt: Prompt | None) -> None:
        controller = self._template_preview_controller_supplier()
        if controller is None:
            return
        controller.update_preview(prompt)


__all__ = ["WorkspaceHistoryController"]
