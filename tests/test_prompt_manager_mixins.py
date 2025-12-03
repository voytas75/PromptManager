"""Unit tests for backend and LiteLLM wiring mixins.

Updates:
  v0.1.0 - 2025-12-03 - Cover backend bootstrap and LiteLLM helper mixins.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from core.embedding import EmbeddingSyncWorker
from core.intent_classifier import IntentClassifier
from core.notifications import NotificationCenter
from core.prompt_manager.backends import NullEmbeddingWorker
from core.prompt_manager.bootstrap import BackendBootstrapMixin
from core.prompt_manager.litellm_helpers import LiteLLMWiringMixin
from core.prompt_manager.workflows import LiteLLMWorkflowMixin

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from core.execution import CodexExecutor
    from core.name_generation import LiteLLMNameGenerator


class _BackendHarness(BackendBootstrapMixin):
    def __init__(self) -> None:
        self._persist_embedding_from_worker = self._noop_persist

    def configure_backends(self, **kwargs: Any) -> None:
        self._initialise_backends(**kwargs)

    def _noop_persist(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def repository(self):
        return self._repository

    @property
    def embedding_worker(self):
        return self._embedding_worker

    @property
    def notification_center(self) -> NotificationCenter:
        return self._notification_center


class _StubWorker(NullEmbeddingWorker):
    """Stub worker used for type-compatible dependency injection in tests."""


def _stub_chroma(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid hitting real Chroma client during mixin tests."""

    class _FakeClient:
        def close(self) -> None:
            return None

    class _FakeCollection:
        def count(self) -> int:  # pragma: no cover - unused helper
            return 0

    def _build_chroma_client(*_args: Any, **_kwargs: Any) -> tuple[_FakeClient, _FakeCollection]:
        return _FakeClient(), _FakeCollection()

    monkeypatch.setattr(
        "core.prompt_manager.bootstrap.build_chroma_client",
        _build_chroma_client,
    )


def test_backend_mixin_initialises_embedding_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_chroma(monkeypatch)
    harness = _BackendHarness()
    db_path = tmp_path / "prompts.db"
    chroma_dir = tmp_path / "chroma"
    harness.configure_backends(
        chroma_path=str(chroma_dir),
        db_path=db_path,
        collection_name="unit-tests",
        cache_ttl_seconds=30,
        redis_client=None,
        chroma_client=None,
        embedding_function=None,
        repository=None,
        category_definitions=None,
        embedding_provider=None,
        embedding_worker=None,
        enable_background_sync=True,
        notification_center=None,
    )

    assert harness.collection_name == "unit-tests"
    assert harness.repository is not None
    assert isinstance(harness.embedding_worker, EmbeddingSyncWorker)

    harness.embedding_worker.stop()


def test_backend_mixin_respects_custom_worker_and_notifications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_chroma(monkeypatch)
    harness = _BackendHarness()
    db_path = tmp_path / "prompts.db"
    chroma_dir = tmp_path / "chroma"
    custom_worker = _StubWorker()
    custom_center = NotificationCenter()

    harness.configure_backends(
        chroma_path=str(chroma_dir),
        db_path=db_path,
        collection_name="unit-tests",
        cache_ttl_seconds=30,
        redis_client=None,
        chroma_client=None,
        embedding_function=None,
        repository=None,
        category_definitions=None,
        embedding_provider=None,
        embedding_worker=custom_worker,
        enable_background_sync=False,
        notification_center=custom_center,
    )

    assert harness.embedding_worker is custom_worker
    assert harness.notification_center is custom_center

    harness.configure_backends(
        chroma_path=str(chroma_dir),
        db_path=db_path,
        collection_name="unit-tests",
        cache_ttl_seconds=30,
        redis_client=None,
        chroma_client=None,
        embedding_function=None,
        repository=harness.repository,
        category_definitions=None,
        embedding_provider=None,
        embedding_worker=None,
        enable_background_sync=False,
        notification_center=None,
    )

    assert isinstance(harness.embedding_worker, NullEmbeddingWorker)


class _LiteHarness(LiteLLMWiringMixin, LiteLLMWorkflowMixin):
    def __init__(self) -> None:
        self._prompt_templates: dict[str, str] = {}
        self._litellm_workflow_models: dict[str, str] = {}

    def configure_helpers(self, **kwargs: Any) -> None:
        self._initialise_litellm_helpers(**kwargs)

    @property
    def litellm_fast_model(self) -> str | None:
        return self._litellm_fast_model

    @property
    def litellm_inference_model(self) -> str | None:
        return self._litellm_inference_model

    @property
    def litellm_workflow_models(self) -> dict[str, str]:
        return self._litellm_workflow_models

    @property
    def prompt_templates(self) -> dict[str, str]:
        return self._prompt_templates

    @property
    def intent_classifier(self) -> IntentClassifier | None:
        return self._intent_classifier


class _Helper:
    def __init__(
        self,
        *,
        model: str | None = None,
        drop_params: tuple[str, ...] | None = None,
        stream: bool | None = None,
    ) -> None:
        self.model = model
        if drop_params is not None:
            self.drop_params = drop_params
        if stream is not None:
            self.stream = stream


def test_litellm_mixin_configures_models_and_executor() -> None:
    harness = _LiteHarness()
    name_helper = cast(
        "LiteLLMNameGenerator",
        _Helper(model="fast-model", drop_params=("alpha", "beta"), stream=True),
    )
    executor = cast(
        "CodexExecutor",
        SimpleNamespace(drop_params=[], reasoning_effort=None, stream=False),
    )

    harness.configure_helpers(
        name_generator=name_helper,
        description_generator=None,
        scenario_generator=None,
        category_generator=None,
        prompt_engineer=None,
        structure_prompt_engineer=None,
        fast_model=None,
        inference_model="slow-model",
        workflow_models={"prompt_execution": "inference"},
        executor=executor,
        intent_classifier=None,
        prompt_templates={"prompt_engineering": " custom "},
    )

    assert harness.litellm_fast_model == "fast-model"
    assert harness.litellm_inference_model == "slow-model"
    assert harness.litellm_workflow_models == {"prompt_execution": "inference"}
    assert executor.drop_params == ["alpha", "beta"]
    assert executor.stream is True
    assert harness.prompt_templates == {"prompt_engineering": "custom"}
    assert isinstance(harness.intent_classifier, IntentClassifier)


def test_litellm_mixin_uses_supplied_intent_classifier() -> None:
    harness = _LiteHarness()
    executor = cast(
        "CodexExecutor",
        SimpleNamespace(drop_params=[], reasoning_effort="medium", stream=False),
    )
    classifier = IntentClassifier()

    harness.configure_helpers(
        name_generator=None,
        description_generator=None,
        scenario_generator=None,
        category_generator=None,
        prompt_engineer=None,
        structure_prompt_engineer=None,
        fast_model="fast",
        inference_model=None,
        workflow_models={"unknown": "inference"},
        executor=executor,
        intent_classifier=classifier,
        prompt_templates=None,
    )

    assert harness.intent_classifier is classifier
    assert harness.litellm_fast_model == "fast"
    assert harness.litellm_workflow_models == {}
    assert executor.reasoning_effort == "medium"
