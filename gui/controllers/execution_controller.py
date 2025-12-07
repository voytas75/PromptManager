"""Coordinate prompt execution, streaming, and chat workspace logic.

Updates:
  v0.1.5 - 2025-12-07 - Support promptless workspace text execution.
  v0.1.4 - 2025-12-07 - Wrap web context with shared markers and numbering.
  v0.1.3 - 2025-12-04 - Include all web snippets and summarize >5k words via fast LiteLLM.
  v0.1.2 - 2025-12-04 - Surface web context source counts before each workspace result.
  v0.1.1 - 2025-12-04 - Add optional web search enrichment controlled by a UI toggle.
  v0.1.0 - 2025-11-30 - Extract execution and chat orchestration from main window.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from html import escape
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QTextCursor
from PySide6.QtWidgets import QAbstractButton, QStyle

from config import (
    DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    PromptManagerSettings,
)
from core import (
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    WebSearchError,
)
from core.execution import ExecutionError
from core.litellm_adapter import (
    LiteLLMNotInstalledError,
    apply_configured_drop_params,
    call_completion_with_fallback,
    get_completion,
    serialise_litellm_response,
)
from core.web_search.context_formatting import (
    build_numbered_search_results,
    wrap_search_results_block,
)

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from uuid import UUID

    from PySide6.QtWidgets import (
        QAbstractButton,
        QCheckBox,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QTabWidget,
        QTextEdit,
    )

from gui.voice_playback_controller import VoicePlaybackError
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from gui.share_controller import ShareController
    from gui.usage_logger import IntentUsageLogger
    from gui.voice_playback_controller import VoicePlaybackController

StatusCallback = Callable[[str, int], None]
ClearStatusCallback = Callable[[], None]
ErrorCallback = Callable[[str, str], None]
ToastCallback = Callable[[str, int], None]


WEB_SEARCH_RESULT_LIMIT = 10
WEB_CONTEXT_SUMMARY_WORD_LIMIT = 5000
WEB_CONTEXT_SUMMARY_SYSTEM_PROMPT = (
    "You are a fast research assistant who condenses multiple web search excerpts "
    "into a concise, factual briefing for another model. Highlight consensus, note "
    "conflicts, and reference the provided source URLs inline when helpful. Respond "
    "in no more than 400 words."
)

FREEFORM_PROMPT_NAME = "Workspace Text Run"
FREEFORM_PROMPT_CATEGORY = "Workspace"
FREEFORM_SYSTEM_PROMPT = (
    "You are a versatile AI assistant. Treat the user's next message as a standalone request "
    "and respond with the clearest, most actionable answer possible."
)

logger = logging.getLogger(__name__)


class ExecutionController:
    """Manage workspace execution, streaming output, and chat state."""

    def __init__(
        self,
        *,
        manager: PromptManager,
        runtime_settings: dict[str, object | None],
        usage_logger: IntentUsageLogger,
        share_controller: ShareController,
        query_input: QPlainTextEdit,
        result_label: QLabel,
        result_meta: QLabel,
        result_tabs: QTabWidget,
        result_text: QTextEdit,
        chat_history_view: QTextEdit,
        render_markdown_checkbox: QCheckBox | None,
        copy_result_button: QPushButton,
        copy_result_to_text_window_button: QPushButton,
        save_button: QPushButton,
        share_result_button: QPushButton,
        speak_result_button: QAbstractButton,
        web_search_checkbox: QCheckBox | None,
        continue_chat_button: QPushButton,
        end_chat_button: QPushButton,
        status_callback: StatusCallback,
        clear_status_callback: ClearStatusCallback,
        error_callback: ErrorCallback,
        toast_callback: ToastCallback,
        voice_controller: VoicePlaybackController | None,
        settings: PromptManagerSettings | None = None,
    ) -> None:
        """Store shared dependencies and references to workspace widgets."""
        self._manager = manager
        self._runtime_settings = runtime_settings
        self._usage_logger = usage_logger
        self._share_controller = share_controller
        self._query_input = query_input
        self._result_label = result_label
        self._result_meta = result_meta
        self._result_tabs = result_tabs
        self._result_text = result_text
        self._chat_history_view = chat_history_view
        self._render_markdown_checkbox = render_markdown_checkbox
        self._copy_result_button = copy_result_button
        self._copy_result_to_text_window_button = copy_result_to_text_window_button
        self._save_button = save_button
        self._share_result_button = share_result_button
        self._speak_result_button = speak_result_button
        self._web_search_checkbox = web_search_checkbox
        self._continue_chat_button = continue_chat_button
        self._end_chat_button = end_chat_button
        self._status = status_callback
        self._clear_status = clear_status_callback
        self._error = error_callback
        self._toast = toast_callback
        self._settings = settings
        self._voice_controller = voice_controller
        self._voice_supported = bool(voice_controller and voice_controller.is_supported)
        self._speak_result_button_play_icon = speak_result_button.icon()
        button_style = speak_result_button.style()
        self._speak_result_button_stop_icon = button_style.standardIcon(QStyle.SP_MediaStop)
        self._speak_result_button.setCheckable(True)
        self._web_search_warning_shown = False
        self._last_web_context_sources = 0
        self._last_web_context_provider: str | None = None
        self._web_context_preface = ""

        self._streaming_in_progress = False
        self._stream_prompt_id: UUID | None = None
        self._streaming_buffer: list[str] = []
        self._stream_control_state: dict[str, bool] = {}
        self._raw_result_text = ""
        self._last_execution: PromptManager.ExecutionOutcome | None = None
        self._last_prompt_name: str | None = None
        self._chat_conversation: list[dict[str, str]] = []
        self._chat_prompt_id: UUID | None = None

        self._copy_result_button.setEnabled(False)
        self._copy_result_to_text_window_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._share_result_button.setEnabled(False)
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)
        self._speak_result_button.setEnabled(False)

        if self._voice_controller and self._voice_supported:
            self._voice_controller.playback_preparing.connect(self._on_voice_preparing)
            self._voice_controller.playback_started.connect(self._on_voice_started)
            self._voice_controller.playback_finished.connect(self._on_voice_finished)
            self._voice_controller.playback_failed.connect(self._on_voice_failed)
        else:
            tooltip = (
                "Voice playback requires Qt multimedia support and a configured LiteLLM TTS model."
            )
            self._speak_result_button.setToolTip(tooltip)

    @property
    def last_execution(self) -> PromptManager.ExecutionOutcome | None:
        """Return the last successful execution outcome."""
        return self._last_execution

    @property
    def last_prompt_name(self) -> str | None:
        """Return the name of the prompt tied to the last result."""
        return self._last_prompt_name

    @property
    def raw_result_text(self) -> str:
        """Return the raw, unformatted workspace response text."""
        return self._raw_result_text

    def handle_prompt_selection_change(self, prompt_id: UUID | None) -> None:
        """Synchronise chat session state when the prompt selection changes."""
        if prompt_id is None:
            self._reset_chat_session(clear_view=True)
            return
        if self._chat_prompt_id and prompt_id != self._chat_prompt_id:
            self._reset_chat_session()

    def is_streaming(self) -> bool:
        """Return True while a streaming execution is active."""
        return self._streaming_in_progress

    def refresh_rendering(self) -> None:
        """Re-render output and chat views based on the markdown toggle."""
        self._refresh_result_text_display()
        self._refresh_chat_history_view()

    def refresh_chat_history_view(self) -> None:
        """Expose chat refresh so callers avoid touching internal helpers."""
        self._refresh_chat_history_view()

    def notify_share_providers_changed(self) -> None:
        """Re-evaluate whether the share button should be enabled."""
        self._update_result_share_button_state()

    def can_share_result(self) -> bool:
        """Return True when the current result can be published."""
        return self._share_controller.has_providers() and bool(self._raw_result_text.strip())

    def share_result_text(self, provider_name: str) -> None:
        """Publish the latest workspace output via *provider_name*."""
        payload = self._raw_result_text.strip()
        if not payload:
            self._status("Run a prompt to produce output before sharing.", 4000)
            return
        prompt_label = self._last_prompt_name or "Workspace Result"
        self._share_controller.share_payload(
            provider_name,
            payload,
            prompt=None,
            prompt_name=prompt_label,
            indicator_title="Sharing Result",
            error_title="Unable to share result",
        )

    def copy_result_to_clipboard(self) -> None:
        """Copy the most recent execution result into the system clipboard."""
        if self._last_execution is None:
            self._status("Run a prompt to generate a result first.", 3000)
            return
        response = self._last_execution.result.response_text
        if not response:
            self._status("Latest result is empty; nothing to copy.", 3000)
            return
        QGuiApplication.clipboard().setText(response)
        self._toast("Prompt result copied to the clipboard.", 2500)

    def copy_result_to_workspace(self) -> None:
        """Place the latest execution result inside the workspace editor."""
        if self._last_execution is None:
            self._status("Run a prompt to generate a result first.", 3000)
            return
        response = self._last_execution.result.response_text
        if not response:
            self._status("Latest result is empty; nothing to copy.", 3000)
            return
        self._query_input.setPlainText(response)
        cursor = self._query_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._query_input.setTextCursor(cursor)
        self._query_input.setFocus(Qt.ShortcutFocusReason)
        self._toast("Prompt result copied to the text window.", 2500)

    def execute_prompt_with_text(
        self,
        prompt: Prompt,
        request_text: str,
        *,
        status_prefix: str,
        empty_text_message: str,
        keep_text_after: bool,
    ) -> None:
        """Execute *prompt* using *request_text* while handling streaming state."""
        trimmed = request_text.strip()
        if not trimmed:
            self._status(empty_text_message, 4000)
            return

        payload = self._maybe_enrich_request(prompt, trimmed)
        streaming_enabled = self._is_streaming_enabled()
        callback = self._handle_stream_chunk if streaming_enabled else None
        if streaming_enabled:
            self._begin_streaming_run(prompt)
        try:
            outcome = self._manager.execute_prompt(
                prompt.id,
                payload,
                stream=streaming_enabled,
                on_stream=callback,
            )
        except PromptExecutionUnavailable as exc:
            if streaming_enabled:
                self._end_streaming_run()
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._error("Prompt execution unavailable", str(exc))
            self._status("Prompt execution unavailable.", 4000)
            return
        except (PromptExecutionError, PromptManagerError) as exc:
            if streaming_enabled:
                self._end_streaming_run()
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._error("Prompt execution failed", str(exc))
            return
        finally:
            if streaming_enabled and self._streaming_in_progress:
                self._end_streaming_run()

        self._display_execution_result(prompt, outcome)
        if keep_text_after:
            self._query_input.setPlainText(request_text)
            cursor = self._query_input.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._query_input.setTextCursor(cursor)
            self._query_input.setFocus(Qt.ShortcutFocusReason)

        self._usage_logger.log_execute(
            prompt_name=prompt.name,
            success=True,
            duration_ms=outcome.result.duration_ms,
        )
        self._status(
            f"{status_prefix} '{prompt.name}' in {outcome.result.duration_ms} ms.",
            5000,
        )

    def execute_text_only(self, request_text: str) -> None:
        """Execute workspace text without referencing the selected prompt."""
        trimmed = (request_text or "").strip()
        if not trimmed:
            self._status("Type or paste some text before running it.", 4000)
            return
        executor = getattr(self._manager, "executor", None)
        if executor is None:
            self._error("Prompt execution unavailable", "Configure LiteLLM before running text.")
            self._status("Prompt execution unavailable.", 4000)
            return
        prompt = self._build_freeform_prompt()
        payload = self._maybe_enrich_request(prompt, trimmed)
        streaming_enabled = self._is_streaming_enabled()
        callback = self._handle_stream_chunk if streaming_enabled else None
        if streaming_enabled:
            self._begin_streaming_run(prompt)
        try:
            result = executor.execute(
                prompt,
                payload,
                conversation=None,
                stream=streaming_enabled,
                on_stream=callback,
            )
        except ExecutionError as exc:
            if streaming_enabled and self._streaming_in_progress:
                self._end_streaming_run()
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._error("Prompt execution failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - defensive guard for executor failures
            if streaming_enabled and self._streaming_in_progress:
                self._end_streaming_run()
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._error("Prompt execution failed", str(exc))
            return
        finally:
            if streaming_enabled and self._streaming_in_progress:
                self._end_streaming_run()

        conversation = [{"role": "user", "content": trimmed}]
        if result.response_text:
            conversation.append({"role": "assistant", "content": result.response_text})
        outcome = PromptManager.ExecutionOutcome(
            result=result,
            history_entry=None,
            conversation=conversation,
        )
        self._display_execution_result(
            prompt,
            outcome,
            enable_save=False,
            enable_chat=False,
        )
        self._usage_logger.log_execute(
            prompt_name=prompt.name,
            success=True,
            duration_ms=result.duration_ms,
        )
        self._status(
            f"Executed workspace text run in {result.duration_ms} ms.",
            5000,
        )

    def continue_chat(self) -> None:
        """Send the workspace text as a follow-up within the active chat session."""
        if not self._chat_conversation or self._chat_prompt_id is None:
            self._status("Run a prompt to start a chat before continuing.", 4000)
            return

        follow_up = self._query_input.toPlainText().strip()
        if not follow_up:
            self._status("Type a follow-up message before continuing the chat.", 4000)
            return

        try:
            prompt = self._manager.get_prompt(self._chat_prompt_id)
        except (PromptNotFoundError, PromptManagerError) as exc:
            self._error("Prompt unavailable", str(exc))
            self._reset_chat_session()
            return

        payload = self._maybe_enrich_request(prompt, follow_up)
        streaming_enabled = self._is_streaming_enabled()
        callback = self._handle_stream_chunk if streaming_enabled else None
        if streaming_enabled:
            self._begin_streaming_run(prompt)
        try:
            outcome = self._manager.execute_prompt(
                prompt.id,
                payload,
                conversation=self._chat_conversation,
                stream=streaming_enabled,
                on_stream=callback,
            )
        except PromptExecutionUnavailable as exc:
            if streaming_enabled:
                self._end_streaming_run()
            self._continue_chat_button.setEnabled(False)
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._error("Prompt execution unavailable", str(exc))
            return
        except (PromptExecutionError, PromptManagerError) as exc:
            if streaming_enabled:
                self._end_streaming_run()
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._error("Continue chat failed", str(exc))
            return
        finally:
            if streaming_enabled and self._streaming_in_progress:
                self._end_streaming_run()

        self._display_execution_result(prompt, outcome)
        self._usage_logger.log_execute(
            prompt_name=prompt.name,
            success=True,
            duration_ms=outcome.result.duration_ms,
        )
        self._status(
            f"Continued chat with '{prompt.name}' in {outcome.result.duration_ms} ms.",
            5000,
        )

    def end_chat(self) -> None:
        """Terminate the active chat session while preserving the transcript."""
        if not self._chat_conversation:
            self._status("There is no active chat session to end.", 4000)
            return
        self._chat_prompt_id = None
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)
        self._status("Chat session ended. Conversation preserved in history.", 5000)

    def clear_execution_result(self) -> None:
        """Reset all output controls to their default disabled state."""
        self._stop_voice_playback()
        self._last_execution = None
        self._last_prompt_name = None
        self._result_label.setText("No prompt executed yet")
        self._result_meta.clear()
        self._set_result_text_content("")
        self._copy_result_button.setEnabled(False)
        self._copy_result_to_text_window_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._share_result_button.setEnabled(False)
        self._speak_result_button.setEnabled(False)
        self._reset_chat_session()
        self._set_web_context_summary(0, None)

    def abort_streaming(self) -> None:
        """Stop any in-flight streaming run without marking it successful."""
        if self._streaming_in_progress:
            self._end_streaming_run()

    def _is_streaming_enabled(self) -> bool:
        value = self._runtime_settings.get("litellm_stream")
        if value is None and self._settings is not None:
            return bool(getattr(self._settings, "litellm_stream", False))
        return bool(value)

    def _web_search_enabled(self) -> bool:
        checkbox = self._web_search_checkbox
        if checkbox is None:
            return True
        return checkbox.isChecked()

    def _set_web_context_summary(
        self,
        count: int,
        provider_label: str | None,
        *,
        summarized: bool = False,
    ) -> None:
        safe_count = max(0, int(count))
        provider_name = (provider_label or "").strip()
        normalized_provider = provider_name.title() if provider_name else None
        self._last_web_context_sources = safe_count
        self._last_web_context_provider = normalized_provider

        if safe_count == 0 and normalized_provider is None:
            self._web_context_preface = ""
            return

        heading = f"{normalized_provider} context" if normalized_provider else "Web context"
        plural = "source" if safe_count == 1 else "sources"
        summary_note = " (summarized)" if summarized and safe_count else ""
        self._web_context_preface = f"{heading} ({safe_count} {plural} added{summary_note}).\n\n"

    def _build_freeform_prompt(self) -> Prompt:
        return Prompt(
            id=uuid.uuid4(),
            name=FREEFORM_PROMPT_NAME,
            description="Ad-hoc workspace text execution",
            category=FREEFORM_PROMPT_CATEGORY,
            context=FREEFORM_SYSTEM_PROMPT,
            tags=["workspace"],
            language="en",
        )

    def _maybe_enrich_request(self, prompt: Prompt, request_text: str) -> str:
        self._set_web_context_summary(0, None)
        if not request_text or not self._web_search_enabled():
            return request_text
        service = getattr(self._manager, "web_search_service", None)
        if service is None or not getattr(service, "is_available", lambda: False)():
            if not self._web_search_warning_shown:
                self._toast("Web search is not configured; running prompt without it.", 3500)
                self._web_search_warning_shown = True
            return request_text
        query = self._build_web_search_query(prompt, request_text)
        if not query:
            return request_text
        self._status("Gathering live web context…", 0)
        try:
            result = asyncio.run(service.search(query, limit=WEB_SEARCH_RESULT_LIMIT))
        except WebSearchError as exc:
            self._toast(f"Web search failed: {exc}", 4000)
            return request_text
        except Exception:
            self._toast("Web search failed unexpectedly.", 4000)
            return request_text
        finally:
            self._clear_status()
        provider_label = (result.provider or "").strip()
        context_lines, total_words = self._collect_web_context_lines(result.documents)
        if not context_lines:
            self._set_web_context_summary(0, provider_label)
            return request_text
        display_label = provider_label.title() if provider_label else "Web search"
        numbered_context = build_numbered_search_results(context_lines)
        if not numbered_context:
            self._set_web_context_summary(0, provider_label)
            return request_text
        context_block = numbered_context
        summarized = False
        if total_words > WEB_CONTEXT_SUMMARY_WORD_LIMIT:
            self._status("Summarizing web context…", 0)
            try:
                summary_text = self._summarize_web_context(context_block)
            finally:
                self._clear_status()
            if summary_text:
                context_block = summary_text
                summarized = True
        formatted_block = wrap_search_results_block(context_block)
        if not formatted_block:
            self._set_web_context_summary(0, provider_label)
            return request_text
        self._set_web_context_summary(
            len(context_lines),
            display_label,
            summarized=summarized,
        )
        return f"{display_label} findings:\n{formatted_block}\n\nUser request:\n{request_text}"

    def _build_web_search_query(self, prompt: Prompt, request_text: str) -> str:
        parts: list[str] = []
        if prompt.name:
            parts.append(prompt.name.strip())
        if prompt.category:
            parts.append(prompt.category.strip())
        if prompt.tags:
            tags = ", ".join(tag for tag in prompt.tags[:3] if tag)
            if tags:
                parts.append(tags)
        description = (prompt.description or "").strip()
        if description:
            parts.append(description[:160])
        context = (prompt.context or "").strip()
        if context:
            parts.append(context[:160])
        text = request_text.strip()
        if text:
            parts.append(text[:200])
        query = " ".join(part for part in parts if part).strip()
        return query[:512]

    def _collect_web_context_lines(
        self,
        documents: Sequence[object],
    ) -> tuple[list[str], int]:
        context_lines: list[str] = []
        total_words = 0
        for doc in documents:
            if doc is None:
                continue
            url = str(getattr(doc, "url", "") or "").strip()
            if not url:
                continue
            snippet = self._extract_document_snippet(doc)
            if not snippet:
                continue
            total_words += len(snippet.split())
            title = str(getattr(doc, "title", "") or "").strip()
            title_prefix = f"{title}: " if title else ""
            context_lines.append(f"{title_prefix}{snippet} (Source: {url})")
        return context_lines, total_words

    @staticmethod
    def _extract_document_snippet(document: object) -> str:
        segments: list[str] = []
        summary = getattr(document, "summary", None)
        if summary:
            summary_text = str(summary).strip()
            if summary_text:
                segments.append(summary_text)
        highlights = getattr(document, "highlights", None) or []
        for entry in highlights:
            highlight_text = str(entry or "").strip()
            if highlight_text:
                segments.append(highlight_text)
        return " ".join(segments).strip()

    def _summarize_web_context(self, context_block: str) -> str | None:
        settings = self._settings
        if settings is None:
            return None
        model = (getattr(settings, "litellm_model", None) or "").strip()
        if not model:
            logger.debug("Fast LiteLLM model is not configured; skipping web context summary")
            return None
        text = context_block.strip()
        if not text:
            return None
        try:
            completion, LiteLLMException = get_completion()
        except LiteLLMNotInstalledError:
            logger.debug("LiteLLM is not installed; skipping web context summary")
            return None
        request: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": WEB_CONTEXT_SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Summarize the following aggregated web search findings so they can be "
                        "prepended to another LLM request:\n\n"
                        f"{text}"
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        api_key = getattr(settings, "litellm_api_key", None)
        if api_key:
            request["api_key"] = api_key
        api_base = getattr(settings, "litellm_api_base", None)
        if api_base:
            request["api_base"] = api_base
        api_version = getattr(settings, "litellm_api_version", None)
        if api_version:
            request["api_version"] = api_version
        drop_params = getattr(settings, "litellm_drop_params", None)
        dropped = apply_configured_drop_params(request, drop_params)
        try:
            response = call_completion_with_fallback(
                request,
                completion,
                LiteLLMException,
                drop_candidates={"max_tokens", "max_output_tokens", "temperature", "timeout"},
                pre_dropped=dropped,
            )
        except LiteLLMException:
            logger.warning("LiteLLM web context summarization failed", exc_info=True)
            return None
        except Exception:
            logger.warning("Unexpected error while summarizing web context", exc_info=True)
            return None
        payload = serialise_litellm_response(response)
        if payload is None:
            return None
        try:
            message = payload["choices"][0]["message"].get("content")  # type: ignore[index]
        except (KeyError, IndexError, AttributeError, TypeError):
            return None
        if not isinstance(message, str):
            return None
        summary = message.strip()
        return summary or None

    def _begin_streaming_run(self, prompt: Prompt) -> None:
        self._stop_voice_playback()
        self._streaming_in_progress = True
        self._stream_prompt_id = prompt.id
        self._streaming_buffer = []
        self._stream_control_state = {
            "copy": self._copy_result_button.isEnabled(),
            "copy_window": self._copy_result_to_text_window_button.isEnabled(),
            "save": self._save_button.isEnabled(),
            "share": self._share_result_button.isEnabled(),
            "continue": self._continue_chat_button.isEnabled(),
            "end": self._end_chat_button.isEnabled(),
        }
        self._result_label.setText(f"Streaming — {prompt.name}")
        self._result_meta.setText("Receiving response…")
        self._set_result_text_content(self._web_context_preface)
        self._copy_result_button.setEnabled(False)
        self._copy_result_to_text_window_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._share_result_button.setEnabled(False)
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)
        self._update_voice_button_state()
        self._status(f"Streaming '{prompt.name}'…", 0)

    def _handle_stream_chunk(self, chunk: str) -> None:
        if not self._streaming_in_progress or not chunk:
            return
        self._append_result_stream_chunk(chunk)
        self._streaming_buffer.append(chunk)
        QGuiApplication.processEvents()

    def _end_streaming_run(self) -> None:
        self._streaming_in_progress = False
        self._stream_prompt_id = None
        self._streaming_buffer = []
        if self._stream_control_state:
            self._copy_result_button.setEnabled(self._stream_control_state.get("copy", False))
            self._copy_result_to_text_window_button.setEnabled(
                self._stream_control_state.get("copy_window", False)
            )
            self._save_button.setEnabled(self._stream_control_state.get("save", False))
            self._share_result_button.setEnabled(self._stream_control_state.get("share", False))
            self._continue_chat_button.setEnabled(self._stream_control_state.get("continue", False))
            self._end_chat_button.setEnabled(self._stream_control_state.get("end", False))
        else:
            has_result = bool(self._last_execution and self._last_execution.result.response_text)
            self._copy_result_button.setEnabled(has_result)
            self._copy_result_to_text_window_button.setEnabled(has_result)
            self._save_button.setEnabled(self._last_execution is not None)
            self._share_result_button.setEnabled(self._can_share_result())
            self._continue_chat_button.setEnabled(self._last_execution is not None)
            self._end_chat_button.setEnabled(bool(self._chat_conversation))
        self._stream_control_state = {}
        self._clear_status()
        self._update_voice_button_state()

    def _display_execution_result(
        self,
        prompt: Prompt,
        outcome: PromptManager.ExecutionOutcome,
        *,
        enable_save: bool = True,
        enable_chat: bool = True,
    ) -> None:
        self._last_execution = outcome
        self._last_prompt_name = prompt.name
        self._chat_prompt_id = prompt.id if enable_chat else None
        self._chat_conversation = [dict(message) for message in outcome.conversation]
        self._result_label.setText(f"Last Result — {prompt.name}")
        meta_parts: list[str] = [f"Duration: {outcome.result.duration_ms} ms"]
        history_entry = outcome.history_entry
        if history_entry is not None:
            executed_at = history_entry.executed_at.astimezone()
            meta_parts.append(f"Logged: {executed_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self._result_meta.setText(" | ".join(meta_parts))
        response_body = outcome.result.response_text or ""
        response_text = f"{self._web_context_preface}{response_body}".rstrip()
        self._set_result_text_content(response_text)
        self._result_tabs.setCurrentIndex(0)
        has_response = bool(response_text)
        self._copy_result_button.setEnabled(has_response)
        self._copy_result_to_text_window_button.setEnabled(has_response)
        self._save_button.setEnabled(enable_save)
        self._share_result_button.setEnabled(self._can_share_result())
        can_continue = enable_chat and bool(outcome.conversation)
        self._continue_chat_button.setEnabled(can_continue)
        self._end_chat_button.setEnabled(can_continue)
        self._refresh_chat_history_view()
        self._query_input.clear()
        self._query_input.setFocus(Qt.ShortcutFocusReason)

    def _reset_chat_session(self, *, clear_view: bool = True) -> None:
        self._chat_prompt_id = None
        self._chat_conversation = []
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)
        if clear_view:
            self._chat_history_view.clear()
            self._chat_history_view.setPlaceholderText("Run a prompt to start chatting.")

    def _append_result_stream_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._raw_result_text += chunk
        if self._result_text is None:
            return
        if self._is_markdown_render_enabled():
            self._result_text.setMarkdown(self._raw_result_text)
            cursor = self._result_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._result_text.setTextCursor(cursor)
            return
        cursor = self._result_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(chunk)
        self._result_text.setTextCursor(cursor)

    def _refresh_result_text_display(self) -> None:
        if self._result_text is None:
            return
        content = self._raw_result_text
        if self._is_markdown_render_enabled():
            self._result_text.setMarkdown(content)
        else:
            self._result_text.setPlainText(content)
        cursor = self._result_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._result_text.setTextCursor(cursor)

    def _set_result_text_content(self, text: str) -> None:
        self._raw_result_text = text or ""
        self._refresh_result_text_display()
        self._update_result_share_button_state()

    def _update_result_share_button_state(self) -> None:
        if self._streaming_in_progress:
            self._share_result_button.setEnabled(False)
            return
        can_share = self._share_controller.has_providers() and bool(self._raw_result_text.strip())
        self._share_result_button.setEnabled(can_share)

    def _refresh_chat_history_view(self) -> None:
        if not self._chat_conversation:
            self._chat_history_view.clear()
            self._chat_history_view.setPlaceholderText("Run a prompt to start chatting.")
            return
        if self._is_markdown_render_enabled():
            markdown_transcript = self._format_chat_history_markdown(self._chat_conversation)
            self._chat_history_view.setMarkdown(markdown_transcript or "_No messages yet._")
            return
        formatted = self._format_chat_history_html(self._chat_conversation)
        self._chat_history_view.setHtml(formatted)

    def _format_chat_history_html(self, conversation: Sequence[dict[str, str]]) -> str:
        blocks: list[str] = []
        user_colour = self._chat_user_colour()
        user_border = QColor(user_colour).darker(115).name()
        for message in conversation:
            role = message.get("role", "").strip().lower()
            content = message.get("content", "")
            if role == "user":
                speaker = "You"
                bubble_style = (
                    f"background-color: {user_colour}; border: 1px solid {user_border}; "
                    "border-radius: 8px; padding: 8px;"
                )
            elif role == "assistant":
                speaker = "Assistant"
                assistant_colour = self._chat_assistant_colour()
                assistant_border = QColor(assistant_colour).darker(115).name()
                bubble_style = (
                    f"background-color: {assistant_colour}; border: 1px solid {assistant_border}; "
                    "border-radius: 8px; padding: 8px;"
                )
            elif role == "system":
                speaker = "System"
                bubble_style = "background-color: #f0f4f8; border-radius: 8px; padding: 8px;"
            else:
                speaker = message.get("role", "Message")
                bubble_style = "background-color: #f6f7fb; border-radius: 8px; padding: 8px;"

            escaped_content = escape(content).replace("\n", "<br>")
            block = (
                '<div style="margin-bottom: 12px;">'
                f'<div style="font-weight: 600; color: #1f2933; '
                f'margin-bottom: 4px;">{escape(speaker)}:</div>'
                f'<div style="{bubble_style} white-space: pre-wrap; '
                f'line-height: 1.45;">{escaped_content}</div>'
                "</div>"
            )
            blocks.append(block)
        return "<div>" + "".join(blocks) + "</div>"

    def _format_chat_history_markdown(self, conversation: Sequence[dict[str, str]]) -> str:
        blocks: list[str] = []
        for message in conversation:
            role = message.get("role", "").strip().lower()
            content = (message.get("content", "") or "").strip()
            if role == "user":
                speaker = "You"
            elif role == "assistant":
                speaker = "Assistant"
            elif role == "system":
                speaker = "System"
            else:
                speaker = message.get("role", "Message").strip() or "Message"
            if not content:
                content = "_(no content)_"
            block = f"**{speaker}:**\n\n{content}"
            blocks.append(block)
        separator = "\n\n---\n\n"
        return separator.join(blocks)

    def _chat_user_colour(self) -> str:
        fallback = DEFAULT_CHAT_USER_BUBBLE_COLOR
        value = self._runtime_settings.get("chat_user_bubble_color")
        if isinstance(value, str):
            candidate = QColor(value)
            if candidate.isValid():
                fallback = candidate.name().lower()
        return self._chat_palette_colour("user", fallback)

    def _chat_assistant_colour(self) -> str:
        return self._chat_palette_colour("assistant", DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR)

    def _chat_palette_colour(self, role: str, fallback: str) -> str:
        palette_value = self._runtime_settings.get("chat_colors")
        if isinstance(palette_value, dict):
            candidate_value = palette_value.get(role)
            if isinstance(candidate_value, str):
                candidate = QColor(candidate_value)
                if candidate.isValid():
                    return candidate.name().lower()
        fallback_colour = QColor(fallback)
        return fallback_colour.name().lower() if fallback_colour.isValid() else fallback

    def _is_markdown_render_enabled(self) -> bool:
        checkbox = self._render_markdown_checkbox
        return bool(checkbox is None or checkbox.isChecked())

    def _can_share_result(self) -> bool:
        if self._streaming_in_progress:
            return False
        return self._share_controller.has_providers() and bool(self._raw_result_text.strip())

    def _tts_stream_enabled(self) -> bool:
        value = self._runtime_settings.get("litellm_tts_stream")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"false", "0", "no", "off"}:
                return False
            if lowered in {"true", "1", "yes", "on"}:
                return True
        if self._settings is not None:
            return bool(getattr(self._settings, "litellm_tts_stream", True))
        return True

    def toggle_voice_playback(self) -> None:
        """Toggle LiteLLM-powered voice playback for the current result."""
        if not self._voice_controller or not self._voice_supported:
            self._status("Voice playback requires Qt multimedia support.", 4000)
            return
        if not self._raw_result_text.strip():
            self._status("Run a prompt to generate output before using voice playback.", 4000)
            return
        if self._voice_controller.is_active:
            self._voice_controller.stop()
            return
        try:
            self._voice_controller.play_text(
                self._raw_result_text,
                self._runtime_settings,
                stream_audio=self._tts_stream_enabled(),
            )
        except VoicePlaybackError as exc:
            self._status(str(exc), 5000)

    def _on_voice_preparing(self) -> None:
        self._speak_result_button.setEnabled(False)
        self._speak_result_button.setChecked(True)
        self._speak_result_button.setIcon(self._speak_result_button_stop_icon)
        self._status("Generating audio…", 0)

    def _on_voice_started(self) -> None:
        self._speak_result_button.setEnabled(True)
        self._speak_result_button.setChecked(True)
        self._speak_result_button.setIcon(self._speak_result_button_stop_icon)
        self._status("Playing voice output…", 0)

    def _on_voice_finished(self) -> None:
        self._speak_result_button.setChecked(False)
        self._speak_result_button.setIcon(self._speak_result_button_play_icon)
        self._update_voice_button_state()
        self._status("Voice playback finished.", 3000)

    def _on_voice_failed(self, message: str) -> None:
        self._speak_result_button.setChecked(False)
        self._speak_result_button.setIcon(self._speak_result_button_play_icon)
        self._update_voice_button_state()
        self._status(message, 5000)

    def _stop_voice_playback(self) -> None:
        if self._voice_controller and self._voice_controller.is_active:
            self._voice_controller.stop()

    def _update_voice_button_state(self) -> None:
        enabled = (
            self._voice_supported
            and bool(self._raw_result_text.strip())
            and not self._streaming_in_progress
        )
        self._speak_result_button.setEnabled(enabled)
        if not enabled:
            self._speak_result_button.setChecked(False)
            self._speak_result_button.setIcon(self._speak_result_button_play_icon)
