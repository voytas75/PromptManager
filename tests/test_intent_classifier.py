"""Tests for rule-based intent classifier utilities."""

from __future__ import annotations

from core.intent_classifier import IntentClassifier, IntentLabel, rank_by_hints


class _Prompt:
    def __init__(self, name: str, category: str, tags: list[str]) -> None:
        self.id = name
        self.category = category
        self.tags = tags


def test_classifier_detects_debug_intent() -> None:
    classifier = IntentClassifier()
    prediction = classifier.classify("error while running tests in ci")
    assert prediction.label is IntentLabel.DEBUG
    assert prediction.confidence >= 0.35


def test_classifier_detects_refactor_intent() -> None:
    classifier = IntentClassifier()
    prediction = classifier.classify("please refactor this python module for readability")
    assert prediction.label is IntentLabel.REFACTOR
    assert "python" in prediction.language_hints


def test_classifier_defaults_to_general_when_uncertain() -> None:
    classifier = IntentClassifier()
    prediction = classifier.classify("hello there")
    assert prediction.label is IntentLabel.GENERAL


def test_rank_by_hints_prioritises_matching_prompts() -> None:
    prompts = [
        _Prompt("general", "General", ["misc"]),
        _Prompt("debug", "Reasoning / Debugging", ["debugging"]),
        _Prompt("docs", "Documentation", ["docs"]),
    ]

    ranked = rank_by_hints(prompts, category_hints=["Reasoning / Debugging"], tag_hints=["docs"])
    assert [prompt.id for prompt in ranked] == ["debug", "docs", "general"]

