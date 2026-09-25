"""Provider-free regression for benchmark CLI text formatting."""

from __future__ import annotations

import argparse
import logging
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from cli.commands import run_benchmark

if TYPE_CHECKING:
    import pytest


def test_benchmark_text_preserves_token_usage_and_preview(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The benchmark command formats a fake run without invoking any provider."""
    prompt_id = uuid.uuid4()
    run = SimpleNamespace(
        prompt_name="CI triage",
        model="offline-stub",
        error=None,
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        duration_ms=120,
        response_preview="Checked the first failed step.",
        history=None,
    )

    def fake_benchmark_prompts(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(runs=[run])

    manager = SimpleNamespace(benchmark_prompts=fake_benchmark_prompts)
    args = argparse.Namespace(
        prompt_ids=[str(prompt_id)],
        request="Inspect the failed step",
        request_file=None,
        history_window=None,
        trend_window=5,
        models=None,
        persist_history=False,
    )

    assert run_benchmark(cast("Any", manager), args, logging.getLogger(__name__)) == 0
    assert capsys.readouterr().out == (
        "\nBenchmark results\n-----------------\n"
        "- CI triage [offline-stub] -> OK: 120 ms, "
        "tokens(prompt=3, completion=5, total=8)\n"
        "  preview: Checked the first failed step.\n"
    )
