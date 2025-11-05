"""Tests for CodexExecutor streaming behaviour."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable

import pytest

from core.execution import CodexExecutor, ExecutionError
from models.prompt_model import Prompt


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
