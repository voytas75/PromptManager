"""Tests for prompt engineering helpers."""
from __future__ import annotations

import json
from typing import Any

import pytest

from core.prompt_engineering import PromptEngineer, PromptEngineeringError


def test_prompt_engineer_refine_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: dict[str, Any] = {}

    def _fake_completion(**kwargs):
        nonlocal captured_request
        captured_request = kwargs
        payload = {
            "analysis": "Prompt rewritten for clarity.",
            "improved_prompt": "Improved prompt text",
            "checklist": ["Clarity ensured"],
            "warnings": ["Provide API key"],
            "confidence": 0.8,
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(
        "core.prompt_engineering.get_completion",
        lambda: (_fake_completion, Exception),
    )

    engineer = PromptEngineer(model="gpt-4o-mini", temperature=0.1)
    result = engineer.refine(
        "Original prompt",
        name="Prompt Name",
        description="Helpful description",
        category="Analysis",
        tags=["analysis", "debug"],
    )

    assert result.improved_prompt == "Improved prompt text"
    assert result.analysis.startswith("Prompt rewritten")
    assert result.checklist == ["Clarity ensured"]
    assert result.warnings == ["Provide API key"]
    assert 0.79 < result.confidence < 0.81
    assert captured_request["model"] == "gpt-4o-mini"
    assert captured_request["top_p"] == pytest.approx(0.9)


def test_prompt_engineer_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_completion(**kwargs):  # noqa: ARG001
        return {"choices": [{"message": {"content": "not-json"}}]}

    monkeypatch.setattr(
        "core.prompt_engineering.get_completion",
        lambda: (_fake_completion, Exception),
    )

    engineer = PromptEngineer(model="gpt-4o-mini")
    with pytest.raises(PromptEngineeringError):
        engineer.refine("Example prompt")


def test_prompt_engineer_handles_code_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "analysis": "All good",
        "improved_prompt": "Improved",
        "checklist": [],
        "warnings": [],
        "confidence": 0.5,
    }

    def _fake_completion(**kwargs):  # noqa: ARG001
        content = "```json\n" + json.dumps(payload) + "\n```"
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(
        "core.prompt_engineering.get_completion",
        lambda: (_fake_completion, Exception),
    )

    engineer = PromptEngineer(model="gpt-4o-mini")
    result = engineer.refine("Prompt text")

    assert result.improved_prompt == "Improved"


def test_prompt_engineer_supports_object_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "analysis": "ok",
        "improved_prompt": "Improved prompt",
        "checklist": [],
        "warnings": [],
        "confidence": 0.7,
    }

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = {"content": content}

    class _Response:
        def __init__(self, content: str) -> None:
            self.choices = [_Choice(content)]

    def _fake_completion(**kwargs):  # noqa: ARG001
        return _Response(json.dumps(payload))

    monkeypatch.setattr(
        "core.prompt_engineering.get_completion",
        lambda: (_fake_completion, Exception),
    )

    engineer = PromptEngineer(model="gpt-4o-mini")
    result = engineer.refine("Prompt text")

    assert result.improved_prompt == "Improved prompt"


def test_prompt_engineer_supports_structured_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "analysis": "analysis",
        "improved_prompt": "Improved structured",
        "checklist": ["clarity"],
        "warnings": [],
        "confidence": 0.6,
    }

    class _ContentPart:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Message:
        def __init__(self, parts: list[_ContentPart]) -> None:
            self.content = parts

    class _Choice:
        def __init__(self, message: _Message) -> None:
            self.message = message

    class _Response:
        def __init__(self, choice: _Choice) -> None:
            self.choices = [choice]

    def _fake_completion(**kwargs):  # noqa: ARG001
        part = _ContentPart(json.dumps(payload))
        message = _Message([part])
        return _Response(_Choice(message))

    monkeypatch.setattr(
        "core.prompt_engineering.get_completion",
        lambda: (_fake_completion, Exception),
    )

    engineer = PromptEngineer(model="gpt-4o-mini")
    result = engineer.refine("Prompt text")

    assert result.improved_prompt == "Improved structured"


def test_prompt_engineer_parses_json_with_preamble(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "analysis": "analysis",
        "improved_prompt": "Improved after notes",
        "checklist": ["clarity"],
        "warnings": ["Need more context"],
        "confidence": 0.55,
    }

    response_text = (
        "Notes:\n- Ensure coverage\n\nRefined payload follows:\n"
        f"{json.dumps(payload)}\nThanks!"
    )

    def _fake_completion(**kwargs):  # noqa: ARG001
        return {"choices": [{"message": {"content": response_text}}]}

    monkeypatch.setattr(
        "core.prompt_engineering.get_completion",
        lambda: (_fake_completion, Exception),
    )

    engineer = PromptEngineer(model="gpt-4o-mini")
    result = engineer.refine("Prompt text")

    assert result.improved_prompt == "Improved after notes"
    assert result.warnings == ["Need more context"]


def test_prompt_engineer_error_includes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        def __init__(self) -> None:
            self.data = "no choices"

    def _fake_completion(**kwargs):  # noqa: ARG001
        return _Response()

    monkeypatch.setattr(
        "core.prompt_engineering.get_completion",
        lambda: (_fake_completion, Exception),
    )

    engineer = PromptEngineer(model="gpt-4o-mini")
    with pytest.raises(PromptEngineeringError) as exc:
        engineer.refine("Prompt text")
    assert "Payload" in str(exc.value)
