"""Provider-free regression for benchmark CLI text formatting."""

from __future__ import annotations

import argparse
import logging
import uuid
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cli.commands import run_benchmark


@pytest.mark.parametrize("preview", ["Checked the first failed step.", ""])
def test_benchmark_text_preserves_token_usage_and_preview(
    capsys: pytest.CaptureFixture[str],
    preview: str,
) -> None:
    """The benchmark command formats a fake run without invoking any provider."""
    prompt_id = uuid.uuid4()
    run = SimpleNamespace(
        prompt_name="CI triage",
        model="offline-stub",
        error=None,
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        duration_ms=120,
        response_preview=preview,
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
        f"  preview: {preview or '(empty response)'}\n"
    )


@pytest.mark.parametrize(
    "errors",
    [[None, None], [None, "Synthetic failure"], ["First failure", "Second failure"], [""], []],
)
def test_benchmark_domain_exit_preserves_run_diagnostics(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    errors: list[str | None],
) -> None:
    calls: list[str] = []
    runs = [
        SimpleNamespace(
            prompt_name=f"Synthetic {i}",
            model="offline-stub",
            error=error,
            usage={},
            duration_ms=1,
            response_preview="Synthetic result",
            history=None,
        )
        for i, error in enumerate(errors)
    ]

    def benchmark(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        calls.append("benchmark")
        return SimpleNamespace(runs=runs)

    args = argparse.Namespace(prompt_ids=[str(uuid.uuid4())], request="Synthetic input")
    logger = logging.getLogger(__name__)
    with caplog.at_level(logging.INFO, logger=logger.name):
        exit_code = run_benchmark(
            cast("Any", SimpleNamespace(benchmark_prompts=benchmark)), args, logger
        )
    assert calls == ["benchmark"]
    assert exit_code == (0 if errors and all(error is None for error in errors) else 5)
    text = capsys.readouterr().out
    if not errors:
        assert text == ""
        assert "No benchmark runs were executed." in caplog.text
    else:
        assert text.count("-> OK:") == errors.count(None)
        assert text.count("-> ERROR:") == len(errors) - errors.count(None)
        for error in errors:
            if error:
                assert error in text
