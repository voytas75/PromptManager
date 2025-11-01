"""Tests for LiteLLM name/description generation helpers."""

import pytest

import core.name_generation as name_gen_module
from core.name_generation import (
    LiteLLMNameGenerator,
    LiteLLMDescriptionGenerator,
    NameGenerationError,
    DescriptionGenerationError,
)


def test_litellm_name_generator_returns_trimmed_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_completion(**kwargs):
        assert kwargs["model"] == "gpt-4o-mini"
        assert "api_key" in kwargs and kwargs["api_key"] == "secret"
        return {"choices": [{"message": {"content": "Refactor Navigator"}}]}

    monkeypatch.setattr(
        name_gen_module,
        "get_completion",
        lambda: (_fake_completion, Exception),
    )

    generator = LiteLLMNameGenerator(model="gpt-4o-mini", api_key="secret")
    name = generator.generate("Refactor this service to improve maintainability.")
    assert name == "Refactor Navigator"


def test_litellm_name_generator_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_completion(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        name_gen_module,
        "get_completion",
        lambda: (_failing_completion, RuntimeError),
    )

    generator = LiteLLMNameGenerator(model="gpt-4o-mini")
    with pytest.raises(NameGenerationError):
        generator.generate("Whatever")


def test_litellm_description_generator_returns_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_completion(**kwargs):
        assert kwargs["model"] == "gpt-4o-mini"
        return {"choices": [{"message": {"content": "Summarise prompts clearly and quickly."}}]}

    monkeypatch.setattr(
        name_gen_module,
        "get_completion",
        lambda: (_fake_completion, Exception),
    )

    generator = LiteLLMDescriptionGenerator(model="gpt-4o-mini")
    summary = generator.generate("Prompt body")
    assert "Summarise" in summary


def test_litellm_description_generator_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_completion(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        name_gen_module,
        "get_completion",
        lambda: (_failing_completion, RuntimeError),
    )

    generator = LiteLLMDescriptionGenerator(model="gpt-4o-mini")
    with pytest.raises(DescriptionGenerationError):
        generator.generate("Prompt body")


def test_content_filter_error_is_summarised(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ContentFilterError(Exception):
        pass

    def _failing_completion(**kwargs):  # noqa: ARG001
        raise _ContentFilterError(
            "AzureException - Error code: 400 - {'error': {'code': 'content_filter'}}"
        )

    monkeypatch.setattr(
        name_gen_module,
        "get_completion",
        lambda: (_failing_completion, _ContentFilterError),
    )

    generator = LiteLLMNameGenerator(model="gpt-4o-mini")
    with pytest.raises(NameGenerationError) as exc:
        generator.generate("Suggest a name")
    assert "Azure OpenAI blocked" in str(exc.value)


def test_description_content_filter_error_is_summarised(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ContentFilterError(Exception):
        pass

    def _failing_completion(**kwargs):  # noqa: ARG001
        raise _ContentFilterError(
            "AzureException - Error code: 400 - {'error': {'code': 'ResponsibleAIPolicyViolation'}}"
        )

    monkeypatch.setattr(
        name_gen_module,
        "get_completion",
        lambda: (_failing_completion, _ContentFilterError),
    )

    generator = LiteLLMDescriptionGenerator(model="gpt-4o-mini")
    with pytest.raises(DescriptionGenerationError) as exc:
        generator.generate("Body text")
    assert "Azure OpenAI blocked" in str(exc.value)
