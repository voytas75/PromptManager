"""Tests for intent workspace usage logging utilities.

Updates:
  v0.1.1 - 2025-12-08 - Guard dynamic loader spec for Pyright strict mode.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from core.intent_classifier import IntentLabel, IntentPrediction


def _load_usage_logger():
    module_path = Path(__file__).resolve().parents[1] / "gui" / "usage_logger.py"
    spec = importlib.util.spec_from_file_location("usage_logger", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load usage_logger module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IntentUsageLogger


def test_usage_logger_writes_jsonl(tmp_path) -> None:
    """Ensure usage logger writes detect/suggest/copy events to JSONL."""
    IntentUsageLogger = _load_usage_logger()
    path = tmp_path / "intent_usage.jsonl"
    logger = IntentUsageLogger(path=path)
    prediction = IntentPrediction(
        label=IntentLabel.ANALYSIS,
        confidence=0.55,
        rationale=None,
        category_hints=["Analysis"],
        tag_hints=["analysis"],
        language_hints=["python"],
    )

    logger.log_detect(prediction=prediction, query_text="Investigate failing tests")
    prompts = [SimpleNamespace(name="Test Analyzer")]
    logger.log_suggest(
        prediction=prediction,
        query_text="Investigate failing tests",
        prompts=prompts,
        fallback_used=False,
    )
    logger.log_copy(prompt_name="Test Analyzer", prompt_has_body=True)

    contents = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 3
    detect, suggest, copy = (json.loads(line) for line in contents)

    assert detect["event"] == "detect"
    assert detect["label"] == "analysis"
    assert detect["query_chars"] == len("Investigate failing tests")

    assert suggest["event"] == "suggest"
    assert suggest["results_count"] == 1
    assert suggest["top_prompts"] == ["Test Analyzer"]

    assert copy["event"] == "copy"
    assert copy["prompt_name"] == "Test Analyzer"
    assert copy["has_body"] is True
