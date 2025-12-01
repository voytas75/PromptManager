"""Tests for CodexExecutor execution helpers."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest

from core.execution import (
    CodexExecutor,
    ExecutionError,
    _extract_completion_text,
    _extract_stream_text,
    _extract_stream_usage,
    _serialise_chunk,
    _supports_reasoning,
)
from models.prompt_model import Prompt

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


def _make_prompt() -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name="Streaming Prompt",
        description="Prompt used to validate streaming support.",
        category="tests",
        context="Provide helpful answers.",
    )


def test_codex_executor_collects_streaming_chunks(monkeypatch) -> None:
    prompt = _make_prompt()
    executor = CodexExecutor(model="gpt-test")
    chunks: list[str] = []

    streaming_payload: list[dict[str, Any]] = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [{"delta": {"content": " world"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6},
        },
    ]

    def fake_completion(**request: Any) -> Iterable[dict[str, Any]]:
        assert request.get("stream") is True
        return iter(streaming_payload)

    def fake_get_completion() -> tuple[Callable[..., Any], type[Exception]]:
        return fake_completion, RuntimeError

    monkeypatch.setattr("core.execution.get_completion", fake_get_completion)

    result = executor.execute(
        prompt,
        "Say hello",
        stream=True,
        on_stream=chunks.append,
    )

    assert chunks == ["Hello", " world"]
    assert result.response_text == "Hello world"
    assert dict(result.usage) == {"prompt_tokens": 4, "completion_tokens": 6}
    assert result.raw_response.get("streamed") is True
    assert result.raw_response.get("choices")[0]["message"]["content"] == "Hello world"


def test_codex_executor_stream_requires_iterable(monkeypatch) -> None:
    prompt = _make_prompt()
    executor = CodexExecutor(model="gpt-test")

    def fake_completion(**request: Any) -> int:
        assert request.get("stream") is True
        return 42

    def fake_get_completion() -> tuple[Callable[..., Any], type[Exception]]:
        return fake_completion, RuntimeError

    monkeypatch.setattr("core.execution.get_completion", fake_get_completion)

    with pytest.raises(ExecutionError) as excinfo:
        executor.execute(prompt, "Say hello", stream=True)

    assert "streaming response is not iterable" in str(excinfo.value)


def test_codex_executor_non_stream_request_builds_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _make_prompt()
    executor = CodexExecutor(
        model="gpt-4.1",
        api_key="secret",
        api_base="https://api.example.com",
        api_version="2024-06-01",
        reasoning_effort="medium",
        drop_params=("api_key",),
        max_output_tokens=256,
        temperature=0.5,
    )
    captured_request: dict[str, Any] = {}

    def fake_get_completion() -> tuple[Callable[..., Any], type[Exception]]:
        return (lambda **_: {"choices": []}), RuntimeError

    def fake_call_completion(
        request: Mapping[str, Any],
        completion: Callable[..., Any],  # noqa: ARG001
        lite_llm_exception: type[Exception],  # noqa: ARG001
        *,
        drop_candidates: Sequence[str],  # noqa: ARG001
        pre_dropped: Sequence[str],
    ) -> Mapping[str, Any]:
        captured_request.update(request)
        captured_request["pre_dropped"] = tuple(pre_dropped)
        return {
            "choices": [{"message": {"content": "Execution succeeded"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    monkeypatch.setattr("core.execution.get_completion", fake_get_completion)
    monkeypatch.setattr("core.execution.call_completion_with_fallback", fake_call_completion)

    result = executor.execute(
        prompt,
        "Run quality checks",
        conversation=[{"role": "assistant", "content": "Prior context"}],
    )

    assert result.response_text == "Execution succeeded"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}
    assert captured_request["model"] == "gpt-4.1"
    assert captured_request["messages"][0]["content"] == prompt.context
    assert captured_request["messages"][-1]["content"] == "Run quality checks"
    assert captured_request["reasoning"] == {"effort": "medium"}
    assert captured_request["pre_dropped"] == ("api_key",)
    assert "api_key" not in captured_request


def test_codex_executor_validates_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _make_prompt()
    executor = CodexExecutor(model="gpt-test")

    def fake_get_completion() -> tuple[Callable[..., Any], type[Exception]]:
        return (lambda **_: None), RuntimeError

    monkeypatch.setattr("core.execution.get_completion", fake_get_completion)

    with pytest.raises(ExecutionError) as excinfo:
        executor.execute(
            prompt,
            "Explain logs",
            conversation=[{"role": "", "content": "Missing role"}],
        )

    assert "missing a role" in str(excinfo.value)


def test_supports_reasoning_marks_reasoning_models() -> None:
    assert _supports_reasoning("gpt-4.1") is True
    assert _supports_reasoning("o1-mini") is True
    assert _supports_reasoning("gpt-3.5") is False


def test_extract_completion_text_handles_multiple_shapes() -> None:
    message_payload = {"choices": [{"message": {"content": "From message"}}]}
    delta_payload = {"choices": [{"delta": {"content": "From delta"}}]}
    text_payload = {"choices": [{"text": "From text"}]}

    assert _extract_completion_text(message_payload) == "From message"
    assert _extract_completion_text(delta_payload) == "From delta"
    assert _extract_completion_text(text_payload) == "From text"

    with pytest.raises(ExecutionError):
        _extract_completion_text({"choices": []})


def test_serialise_chunk_prefers_model_dump_and_dict() -> None:
    class _Chunk:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, Any]:
            return self._payload

    chunk = _Chunk({"choices": []})
    assert _serialise_chunk(chunk) == {"choices": []}

    class _FallbackChunk:
        def __init__(self) -> None:
            self.dict_called = False

        def model_dump(self) -> dict[str, Any]:
            raise RuntimeError("boom")

        def dict(self) -> dict[str, Any]:
            self.dict_called = True
            return {"state": "ok"}

    fallback = _FallbackChunk()
    assert _serialise_chunk(fallback) == {"state": "ok"}


def test_extract_stream_helpers_return_text_and_usage() -> None:
    payload = {
        "choices": [{"delta": {"content": " piece "}}],
        "usage": {"prompt_tokens": 1},
    }
    assert _extract_stream_text(payload) == " piece "
    assert _extract_stream_usage(payload) == {"prompt_tokens": 1}

    empty_payload: Mapping[str, Any] = {}
    assert _extract_stream_text(empty_payload) == ""
    assert _extract_stream_usage(empty_payload) == {}
