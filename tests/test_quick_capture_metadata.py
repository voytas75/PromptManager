"""Offline contracts for Quick Capture automatic metadata."""

from __future__ import annotations

import json
import socket
from typing import TYPE_CHECKING, cast

import pytest

from gui.dialogs.quick_capture import QuickCaptureDraft
from gui.prompt_generation_service import PromptGenerationService

if TYPE_CHECKING:
    from pathlib import Path

    from core import PromptManager


def test_strict_category_tags_use_existing_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from core import PromptManager, litellm_adapter
    from core.name_generation import LiteLLMNameGenerator

    manager = object.__new__(PromptManager)
    monkeypatch.setattr(
        manager,
        "_category_generator",
        LiteLLMNameGenerator(
            model="test-model",
            api_key="test-key",
            api_base="https://invalid.example",
            api_version="test-version",
            drop_params=["temperature"],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        manager, "_name_generator", LiteLLMNameGenerator(model="wrong-name-route"), raising=False
    )
    monkeypatch.setattr(manager, "_llm_available", True, raising=False)
    monkeypatch.setattr(manager, "list_categories", lambda: list[object]())

    def completion(**request: object) -> object:
        assert request["model"] == "test-model"
        assert request["api_key"] == "test-key"
        assert request["api_base"] == "https://invalid.example"
        assert request["api_version"] == "test-version"
        assert "temperature" not in request
        assert request["messages"] == [
            {
                "role": "system",
                "content": cast("list[dict[str, str]]", request["messages"])[0]["content"],
            },
            {"role": "user", "content": "Explain security"},
        ]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"category": "Model category", "tags": [" model-only ", "model-only"]}
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(litellm_adapter, "get_completion", lambda: (completion, RuntimeError))
    assert manager.generate_prompt_category_and_tags("Explain security") == (
        "Model category",
        ["model-only"],
    )


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{}",
        '{"category":"C","tags":[1]}',
        '{"category":"C","tags":[""]}',
        '{"category":"C","tags":[]}',
    ],
)
def test_strict_model_response_rejects_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    from core import PromptManager, PromptManagerError, litellm_adapter
    from core.name_generation import LiteLLMNameGenerator

    manager = object.__new__(PromptManager)
    monkeypatch.setattr(
        manager, "_category_generator", LiteLLMNameGenerator(model="test-model"), raising=False
    )
    monkeypatch.setattr(
        manager, "_name_generator", LiteLLMNameGenerator(model="wrong-name-route"), raising=False
    )
    monkeypatch.setattr(manager, "_llm_available", True, raising=False)
    monkeypatch.setattr(manager, "list_categories", lambda: list[object]())

    def completion(**kwargs: object) -> object:
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(litellm_adapter, "get_completion", lambda: (completion, RuntimeError))
    with pytest.raises(PromptManagerError):
        manager.generate_prompt_category_and_tags("Explain security")


class _Manager:
    llm_available = True

    def generate_prompt_name(self, text: str) -> str:
        assert text == "Explain security"
        return "Security guide"

    def generate_prompt_description(self, text: str, *, allow_fallback: bool = True) -> str:
        assert not allow_fallback
        return "A useful guide"

    def generate_prompt_category(self, text: str) -> str:
        raise AssertionError("Fallback category path forbidden")

    def generate_prompt_category_and_tags(self, text: str) -> tuple[str, list[str]]:
        assert text == "Explain security"
        return "Model category", ["security", "model-only"]

    def generate_prompt_scenarios(self, text: str, *, strict: bool = False) -> list[str]:
        return ["Review security"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _service(manager: _Manager) -> PromptGenerationService:
    service = object.__new__(PromptGenerationService)
    service._manager = cast("PromptManager", manager)  # pyright: ignore[reportPrivateUsage]
    service.generate_prompt_tags = lambda context: pytest.fail("Heuristic tags forbidden")
    return service


@pytest.mark.parametrize("manual", [False, True])
def test_metadata_preserves_body_source_state_and_manual_fields(manual: bool) -> None:
    draft = QuickCaptureDraft(
        body="Explain security",
        title="Mine" if manual else "",
        description="My note" if manual else "",
        tags_text="security, mine",
        source_label="chat",
    )
    original = draft.to_prompt()
    enriched, message = _service(_Manager()).generate_quick_capture_metadata(draft)
    assert enriched.name == ("Mine" if manual else "Security guide")
    assert enriched.description == ("My note" if manual else "A useful guide")
    assert enriched.category == "Model category"
    assert enriched.tags == ["security", "mine", "model-only"]
    assert enriched.scenarios == ["Review security"]
    assert enriched.context == original.context
    assert enriched.source == original.source
    assert enriched.ext2 == original.ext2
    assert "generated" in message


@pytest.mark.parametrize("result", [None, ("", ["tag"]), ("Category", [])])
def test_category_tags_failure_keeps_original_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch, result: tuple[str, list[str]] | None
) -> None:
    def generate(self: _Manager, text: str) -> tuple[str, list[str]]:
        if result is None:
            raise RuntimeError("provider failure")
        return result

    monkeypatch.setattr(_Manager, "generate_prompt_category_and_tags", generate)
    draft = QuickCaptureDraft(body="Explain security", title="Mine", tags_text="mine")
    prompt, message = _service(_Manager()).generate_quick_capture_metadata(draft)
    original = draft.to_prompt()
    assert prompt.name == original.name
    assert prompt.description == original.description
    assert prompt.tags == original.tags
    assert prompt.category == original.category
    assert prompt.scenarios == original.scenarios
    assert prompt.context == original.context
    assert prompt.ext2 == original.ext2
    assert "failed" in message
    assert "generated" not in message


def test_no_model_does_not_call_generators() -> None:
    manager = _Manager()
    manager.llm_available = False
    draft = QuickCaptureDraft(body="Explain security")
    prompt, message = _service(manager).generate_quick_capture_metadata(draft)
    assert prompt.name == draft.to_prompt().name
    assert "no model" in message


def test_failure_keeps_deterministic_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: _Manager, text: str, *, strict: bool = False) -> list[str]:
        raise RuntimeError("provider failure")

    monkeypatch.setattr(_Manager, "generate_prompt_scenarios", fail)
    draft = QuickCaptureDraft(body="Explain security")
    prompt, message = _service(_Manager()).generate_quick_capture_metadata(draft)
    assert prompt.name == draft.to_prompt().name
    assert prompt.scenarios == []
    assert "failed" in message


def test_generated_category_recomputes_slug(tmp_path: Path) -> None:
    prompt, message = _service(_Manager()).generate_quick_capture_metadata(
        QuickCaptureDraft(body="Explain security")
    )
    assert "generated" in message
    assert prompt.category_slug == "model-category"
    from core import PromptManager
    from core.category_registry import CategoryRegistry, load_category_definitions
    from core.repository import PromptRepository

    manager = object.__new__(PromptManager)
    registry = CategoryRegistry(
        PromptRepository(str(tmp_path / "categories.sqlite")), load_category_definitions()
    )
    manager._category_registry = registry  # pyright: ignore[reportPrivateUsage]
    manager._apply_category_metadata(prompt)  # pyright: ignore[reportPrivateUsage]
    assert (prompt.category, prompt.category_slug) == ("Model category", "model-category")


@pytest.mark.parametrize("category", ["!!!", "日本語"])
def test_invalid_category_admission_rolls_back(
    monkeypatch: pytest.MonkeyPatch, category: str
) -> None:
    def generate(self: _Manager, text: str) -> tuple[str, list[str]]:
        return category, ["tag"]

    monkeypatch.setattr(_Manager, "generate_prompt_category_and_tags", generate)
    prompt, message = _service(_Manager()).generate_quick_capture_metadata(
        QuickCaptureDraft(body="Explain security", title="Mine", tags_text="mine")
    )
    assert "failed" in message
    assert prompt.category == "General"
    assert prompt.tags == ["mine"]
    assert prompt.name == "Mine"
    from models.category_model import PromptCategory

    with pytest.raises(ValueError, match="slug cannot be empty"):
        PromptCategory(slug="", label=category, description="")


@pytest.mark.parametrize(
    "content", ["null", "[null]", "[1]", "[true]", "{}", '"scenario"', '[""]', '["-"]', "not json"]
)
def test_real_scenario_parser_strict_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    from core import scenario_generation
    from core.scenario_generation import LiteLLMScenarioGenerator, ScenarioGenerationError

    def completion(**kwargs: object) -> object:
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(scenario_generation, "get_completion", lambda: (completion, RuntimeError))
    generator = LiteLLMScenarioGenerator(model="scenario-route")
    with pytest.raises(ScenarioGenerationError):
        generator.generate("Body", strict=True)

    def generate(self: _Manager, text: str, *, strict: bool = False) -> list[str]:
        assert strict
        return generator.generate(text, strict=strict)

    monkeypatch.setattr(_Manager, "generate_prompt_scenarios", generate)
    prompt, status = _service(_Manager()).generate_quick_capture_metadata(
        QuickCaptureDraft(body="Explain security", title="Mine", tags_text="mine")
    )
    assert "failed" in status
    assert (prompt.name, prompt.category, prompt.tags, prompt.scenarios) == (
        "Mine",
        "General",
        ["mine"],
        [],
    )


def test_real_scenario_parser_strict_and_ordinary_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import scenario_generation
    from core.scenario_generation import LiteLLMScenarioGenerator

    content = '[" Review ", "Review"]'

    def completion(**kwargs: object) -> object:
        assert kwargs["model"] == "scenario-route"
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(scenario_generation, "get_completion", lambda: (completion, RuntimeError))
    generator = LiteLLMScenarioGenerator(model="scenario-route")
    assert generator.generate("Body", strict=True) == ["Review"]
    content = "null"
    assert generator.generate("Body") == ["None"]
