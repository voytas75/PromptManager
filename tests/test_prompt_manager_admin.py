"""Administrative and diagnostic tests for PromptManager."""

from __future__ import annotations

import sqlite3
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest
from chromadb.errors import ChromaError

from core.history_tracker import HistoryTrackerError
from core.intent_classifier import IntentLabel, IntentPrediction
from core.prompt_manager import (
    CategorySuggestionError,
    DescriptionGenerationError,
    NameGenerationError,
    PromptManager,
    PromptManagerError,
    PromptStorageError,
    RedisError,
    ScenarioGenerationError,
)
from core.embedding import EmbeddingGenerationError
from core.exceptions import CategoryError, CategoryStorageError
from core.repository import PromptRepository, RepositoryError
from models.category_model import PromptCategory
from models.prompt_model import ExecutionStatus, Prompt
from prompt_templates import DEFAULT_PROMPT_TEMPLATES


class _DummyCollection:
    """Minimal Chroma collection double for admin tests."""

    def __init__(self) -> None:
        self.count_value = 0
        self.count_exception: Optional[BaseException] = None
        self.peek_calls: List[Dict[str, Any]] = []
        self.upsert_payloads: List[Dict[str, Any]] = []

    def count(self) -> int:
        if self.count_exception:
            raise self.count_exception
        return self.count_value

    def delete(self, **_: Any) -> None:  # noqa: D401 - interface compatibility
        """No-op delete for compatibility."""

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_payloads.append(dict(kwargs))

    def query(self, **_: Any) -> Mapping[str, List[List[str]]]:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

    def peek(self, **kwargs: Any) -> List[Any]:
        self.peek_calls.append(dict(kwargs))
        return []


class _DummyChromaClient:
    """Provide deterministic Chroma client behaviour."""

    def __init__(self, collection: _DummyCollection) -> None:
        self.collection = collection
        self.persist_calls = 0

    def get_or_create_collection(self, **_: Any) -> _DummyCollection:
        return self.collection

    def persist(self) -> None:
        self.persist_calls += 1


class _HistoryTrackerStub:
    """Record execution queries issued by PromptManager."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.should_raise = False

    def query_executions(
        self,
        *,
        status: Optional[ExecutionStatus],
        prompt_id: Optional[uuid.UUID],
        search: Optional[str],
        limit: Optional[int],
    ) -> List[str]:
        if self.should_raise:
            raise HistoryTrackerError("tracker failure")
        payload = {
            "status": status,
            "prompt_id": prompt_id,
            "search": search,
            "limit": limit,
        }
        self.requests.append(payload)
        return ["result"]


class _RedisPoolStub:
    """Expose connection kwargs for Redis diagnostics."""

    def __init__(self, **kwargs: Any) -> None:
        self.connection_kwargs = dict(kwargs)


class _RedisClientStub:
    """Redis client double with controllable responses."""

    def __init__(
        self,
        *,
        ping_result: bool = True,
        ping_error: Optional[BaseException] = None,
        dbsize_value: int = 0,
        info_payload: Optional[Mapping[str, Any]] = None,
        info_error: Optional[BaseException] = None,
        connection_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._ping_result = ping_result
        self._ping_error = ping_error
        self._dbsize_value = dbsize_value
        self._info_payload = dict(info_payload or {})
        self._info_error = info_error
        self.connection_pool: Optional[_RedisPoolStub] = None
        if connection_kwargs is not None:
            self.connection_pool = _RedisPoolStub(**connection_kwargs)

    def ping(self) -> bool:
        if self._ping_error:
            raise self._ping_error
        return self._ping_result

    def dbsize(self) -> int:
        return self._dbsize_value

    def info(self) -> Mapping[str, Any]:
        if self._info_error:
            raise self._info_error
        return dict(self._info_payload)

    def get(self, name: str) -> Optional[bytes]:  # pragma: no cover - unused surface
        return None

    def setex(self, name: str, time: int, value: Any) -> bool:  # pragma: no cover - unused
        return True

    def delete(self, *names: str) -> int:  # pragma: no cover - unused
        return 0

    def close(self) -> None:  # pragma: no cover - unused
        return None


def _make_recorder() -> type:
    """Return a recorder class capturing LiteLLM factory kwargs."""

    class Recorder:
        instances: List["Recorder"] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.drop_params = kwargs.get("drop_params")
            self.stream = kwargs.get("stream", False)
            self.reasoning_effort = kwargs.get("reasoning_effort")
            Recorder.instances.append(self)

    return Recorder


class _ExecutorRecorder:
    """Capture CodexExecutor constructor arguments."""

    instances: List["_ExecutorRecorder"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.drop_params: Optional[Sequence[str]] = kwargs.get("drop_params")
        self.reasoning_effort = kwargs.get("reasoning_effort")
        self.stream = kwargs.get("stream", False)
        _ExecutorRecorder.instances.append(self)


@pytest.fixture()
def prompt_manager(tmp_path: Path) -> tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path]:
    """Fixture providing a PromptManager wired to stub backends."""

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "prompt_manager.db"
    collection = _DummyCollection()
    client = _DummyChromaClient(collection)
    manager = PromptManager(
        chroma_path=str(chroma_dir),
        db_path=str(db_path),
        chroma_client=client,
        enable_background_sync=False,
    )
    return manager, collection, client, chroma_dir


def _ensure_chroma_database(chroma_dir: Path) -> Path:
    """Create or return the Chroma SQLite path used for maintenance operations."""

    db_path = chroma_dir / "chroma.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS diagnostics (id INTEGER PRIMARY KEY)")
    return db_path


def _make_prompt(name: str = "Diagnostics") -> Prompt:
    """Return a prompt instance with predictable metadata."""

    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Investigate regressions",
        category="General",
        context="Analyse telemetry",
    )


def test_set_name_generator_configures_workflows(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _, _ = prompt_manager

    name_factory = _make_recorder()
    description_factory = _make_recorder()
    engineer_factory = _make_recorder()
    scenario_factory = _make_recorder()
    executor_factory = _ExecutorRecorder
    executor_factory.instances.clear()

    monkeypatch.setattr("core.prompt_manager.LiteLLMNameGenerator", name_factory)
    monkeypatch.setattr("core.prompt_manager.LiteLLMDescriptionGenerator", description_factory)
    monkeypatch.setattr("core.prompt_manager.PromptEngineer", engineer_factory)
    monkeypatch.setattr("core.prompt_manager.LiteLLMScenarioGenerator", scenario_factory)
    monkeypatch.setattr("core.prompt_manager.CodexExecutor", executor_factory)

    overrides = {
        "name_generation": " Custom name template ",
        "scenario_generation": "Scenario helper",
        "prompt_engineering": DEFAULT_PROMPT_TEMPLATES["prompt_engineering"],
        "category_generation": 123,  # ignored: not a string
    }

    manager.set_name_generator(
        model="fast-model",
        api_key="secret",
        api_base="https://api.example.com",
        api_version="2024-06-01",
        inference_model="inference-model",
        workflow_models={
            "prompt_execution": "inference",
            "scenario_generation": "inference",
            "unknown_workflow": "fast",
        },
        drop_params=[" api_key ", ""],
        reasoning_effort="medium",
        stream=True,
        prompt_templates=overrides,
    )

    assert manager._litellm_workflow_models == {
        "prompt_execution": "inference",
        "scenario_generation": "inference",
    }
    assert manager._prompt_templates == {
        "name_generation": "Custom name template",
        "scenario_generation": "Scenario helper",
    }
    assert name_factory.instances[0].kwargs["model"] == "fast-model"
    assert scenario_factory.instances[0].kwargs["model"] == "inference-model"
    assert engineer_factory.instances and len(engineer_factory.instances) == 2
    assert executor_factory.instances[0].kwargs["reasoning_effort"] == "medium"
    assert manager._executor is executor_factory.instances[0]
    assert manager._executor.drop_params == ["api_key"]
    assert manager._executor.stream is True


def test_set_name_generator_without_models_resets_state(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    manager._name_generator = object()
    manager._litellm_drop_params = ("api_key",)
    manager._litellm_reasoning_effort = "high"
    manager._litellm_stream = True

    manager.set_name_generator(None, None, None, None)

    assert manager._name_generator is None
    assert manager._litellm_drop_params is None
    assert manager._litellm_reasoning_effort is None
    assert manager._litellm_stream is False


def test_get_redis_details_reports_stats(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    redis_client = _RedisClientStub(
        ping_result=True,
        dbsize_value=42,
        info_payload={
            "used_memory_human": "1M",
            "used_memory_peak_human": "2M",
            "maxmemory_human": "3M",
            "keyspace_hits": "80",
            "keyspace_misses": "20",
            "role": "primary",
        },
        connection_kwargs={
            "host": "localhost",
            "port": 6379,
            "db": 2,
            "username": "cache",
            "ssl": True,
        },
    )
    manager._redis_client = redis_client

    details = manager.get_redis_details()

    assert details["status"] == "online"
    assert details["connection"]["host"] == "localhost"
    assert details["stats"]["hit_rate"] == 80.0
    assert details["role"] == "primary"


def test_get_redis_details_handles_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    redis_client = _RedisClientStub(
        ping_error=RedisError("ping failure"),
        info_error=RedisError("info failure"),
        dbsize_value=4,
    )
    manager._redis_client = redis_client

    details = manager.get_redis_details()
    assert details["status"] == "error"
    assert "ping failure" in details["error"]


def test_get_chroma_details_collects_metrics(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, collection, _, chroma_dir = prompt_manager
    collection.count_value = 3
    data_file = chroma_dir / "collection.bin"
    data_file.write_bytes(b"x" * 10)

    details = manager.get_chroma_details()

    assert details["stats"]["documents"] == 3
    assert details["stats"]["disk_usage_bytes"] >= 10


def test_get_chroma_details_handles_collection_error(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, collection, _, _ = prompt_manager
    collection.count_exception = ChromaError("count failed")

    details = manager.get_chroma_details()
    assert details["status"] == "error"
    assert "count failed" in details["error"]


def test_verify_vector_store_success(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, collection, client, chroma_dir = prompt_manager
    collection.count_value = 2
    _ensure_chroma_database(chroma_dir)

    summary = manager.verify_vector_store()

    assert "SQLite integrity_check: ok" in summary
    assert "Collection count: 2" in summary
    assert collection.peek_calls[0]["limit"] == 2
    assert client.persist_calls >= 1


def test_verify_vector_store_requires_database(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    db_path = _ensure_chroma_database(chroma_dir)
    db_path.unlink()

    with pytest.raises(PromptStorageError):
        manager.verify_vector_store()


def test_compact_and_optimize_vector_store(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    _ensure_chroma_database(chroma_dir)

    manager.compact_vector_store()
    manager.optimize_vector_store()


def test_query_executions_handles_filters(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    tracker = _HistoryTrackerStub()
    manager.set_history_tracker(tracker)

    first = manager.query_executions(status="unknown", search="logs", limit=5)
    second = manager.query_executions(status=ExecutionStatus.SUCCESS)
    tracker.should_raise = True
    fallback = manager.query_executions(status=ExecutionStatus.FAILED)

    assert first == ["result"]
    assert tracker.requests[0]["status"] is None
    assert tracker.requests[1]["status"] == ExecutionStatus.SUCCESS
    assert fallback == []


def test_fallback_category_uses_classifier_hints(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    categories = [
        PromptCategory(slug="reasoning-debugging", label="Reasoning / Debugging", description=""),
        PromptCategory(slug="general", label="General", description=""),
    ]
    prediction = IntentPrediction(
        label=IntentLabel.DEBUG,
        confidence=0.9,
        category_hints=["unknown", "Reasoning / Debugging"],
    )

    class _Classifier:
        def classify(self, _: str) -> IntentPrediction:
            return prediction

    manager.set_intent_classifier(_Classifier())
    assert (
        manager._fallback_category_from_context("Investigate bug", categories) == "Reasoning / Debugging"
    )

    manager.set_intent_classifier(None)
    assert manager._fallback_category_from_context("Document the feature", categories) == "Documentation"


def test_build_description_fallback_composes_segments(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt()
    prompt.tags = ["ops", " observability "]
    prompt.scenarios = [" Investigate regressions "]

    description = manager._build_description_fallback("Context " * 80, prompt)
    assert "focuses on general workflows" in description.lower()
    assert "Common tags: ops, observability." in description
    assert "Example use: Investigate regressions." in description
    assert "Overview" in description

    assert manager._build_description_fallback("", None) == "No description available."


def test_repository_verification_and_maintenance(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    manager.create_prompt(_make_prompt("Repository prompt"))

    summary = manager.verify_repository()
    assert "SQLite integrity_check: ok" in summary

    manager.compact_repository()
    manager.optimize_repository()


def test_generate_prompt_name_and_scenarios(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    with pytest.raises(NameGenerationError):
        manager.generate_prompt_name("Discuss optimisations")
    with pytest.raises(ScenarioGenerationError):
        manager.generate_prompt_scenarios("Discuss optimisations")

    class _NameGenerator:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def generate(self, context: str) -> str:
            self.calls.append(context)
            return "Optimised prompt"

    class _ScenarioGenerator:
        def __init__(self) -> None:
            self.calls: List[tuple[str, int]] = []

        def generate(self, context: str, *, max_scenarios: int) -> List[str]:
            self.calls.append((context, max_scenarios))
            return ["Scenario A"]

    manager._name_generator = _NameGenerator()
    manager._scenario_generator = _ScenarioGenerator()

    assert manager.generate_prompt_name("Discuss optimisations") == "Optimised prompt"
    assert manager.generate_prompt_scenarios("Discuss optimisations", max_scenarios=4) == [
        "Scenario A"
    ]
    assert manager._scenario_generator.calls[0][1] == 4


def test_rebuild_embeddings_counts_success_and_failures(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, collection, _, _ = prompt_manager
    manager.create_prompt(_make_prompt("Successful embedding"))
    manager.create_prompt(_make_prompt("Failing embedding"))

    class _EmbeddingStub:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> List[float]:
            self.calls += 1
            if "Failing" in text:
                raise EmbeddingGenerationError("boom")
            return [0.1, 0.2, 0.3]

    manager._embedding_provider = _EmbeddingStub()  # type: ignore[assignment]
    successes, failures = manager.rebuild_embeddings()

    assert successes == 1
    assert failures == 1
    assert collection.upsert_payloads


def test_suggest_prompts_handles_empty_and_fallbacks(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt_one = manager.create_prompt(_make_prompt("Doc helper one"))
    prompt_two = manager.create_prompt(_make_prompt("Doc helper two"))

    empty_result = manager.suggest_prompts("   ", limit=1)
    assert empty_result.fallback_used is True
    assert empty_result.prompts

    class _Classifier:
        def classify(self, _: str) -> IntentPrediction:
            return IntentPrediction(
                label=IntentLabel.DOCUMENTATION,
                confidence=0.9,
                category_hints=["Documentation"],
                tag_hints=["docs"],
            )

    manager.set_intent_classifier(_Classifier())
    call_counter = {"count": 0}

    def fake_search(self: PromptManager, query_text: str, limit: int = 5) -> List[Prompt]:
        call_counter["count"] += 1
        return [prompt_one] if call_counter["count"] == 1 else [prompt_two]

    manager.search_prompts = types.MethodType(fake_search, manager)
    suggestions = manager.suggest_prompts("Document release", limit=2)

    assert [prompt.name for prompt in suggestions.prompts] == [
        prompt_one.name,
        prompt_two.name,
    ]
    assert suggestions.fallback_used is True


def test_suggest_prompts_requires_positive_limit(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    with pytest.raises(ValueError):
        manager.suggest_prompts("docs", limit=0)


def test_manager_initialises_litellm_defaults_from_helpers(tmp_path: Path) -> None:
    class _GeneratorStub:
        def __init__(self, *, stream: bool = False, drop_params: Optional[Sequence[str]] = None) -> None:
            self.model = "generator-fast"
            self.drop_params = drop_params
            self.stream = stream

    class _ExecutorStub:
        def __init__(self) -> None:
            self.drop_params: List[str] = []
            self.stream = False
            self.reasoning_effort: Optional[str] = None

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "repo.db"
    repository = PromptRepository(str(db_path))
    collection = _DummyCollection()
    client = _DummyChromaClient(collection)
    generator = _GeneratorStub(stream=True, drop_params=("api_key",))
    executor = _ExecutorStub()

    manager = PromptManager(
        chroma_path=str(chroma_dir),
        db_path=str(db_path),
        chroma_client=client,
        repository=repository,
        enable_background_sync=False,
        name_generator=generator,
        scenario_generator=generator,
        prompt_engineer=generator,
        executor=executor,  # type: ignore[arg-type]
        workflow_models={
            "prompt_execution": "inference",
            "unknown": "fast",
        },
    )

    assert manager._litellm_fast_model == "generator-fast"
    assert manager._litellm_workflow_models == {"prompt_execution": "inference"}
    assert manager._litellm_drop_params == ("api_key",)
    assert manager._executor.drop_params == ["api_key"]
    assert manager._litellm_stream is True


def test_refresh_user_profile_handles_repository_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    class _FailingRepo:
        def get_user_profile(self) -> Prompt:
            raise RepositoryError("boom")

    failing_repo = _FailingRepo()
    manager._repository = failing_repo  # type: ignore[assignment]
    cached = manager._user_profile

    assert manager.refresh_user_profile() is cached


def test_generate_prompt_name_and_description_error_paths(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    class _FailingGenerator:
        def generate(self, _: str) -> str:
            raise RuntimeError("fail")

    manager._name_generator = _FailingGenerator()
    with pytest.raises(NameGenerationError):
        manager.generate_prompt_name("Explain telemetry")

    with pytest.raises(DescriptionGenerationError):
        manager.generate_prompt_description("   ")

    summary = manager.generate_prompt_description("Describe behaviour", prompt=_make_prompt())
    assert "Overview" in summary

    manager._description_generator = _FailingGenerator()
    with pytest.raises(DescriptionGenerationError):
        manager.generate_prompt_description("Explain behaviour", allow_fallback=False)


def test_generate_prompt_scenarios_handles_generator_failures(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    class _FailingScenarioGenerator:
        def generate(self, *_: Any, **__: Any) -> List[str]:
            raise RuntimeError("fail")

    manager._scenario_generator = _FailingScenarioGenerator()
    with pytest.raises(ScenarioGenerationError):
        manager.generate_prompt_scenarios("Outline plan")


def test_run_category_generator_and_fallbacks(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    categories = [
        PromptCategory(slug="documentation", label="Documentation", description="Docs"),
        PromptCategory(slug="reporting", label="Reporting", description="Reports"),
    ]

    assert manager._run_category_generator("ctx", categories) == ""

    class _FailingCategoryGenerator:
        def generate(self, *_: Any, **__: Any) -> str:
            raise CategorySuggestionError("fail")

    manager._category_generator = _FailingCategoryGenerator()
    assert manager._run_category_generator("ctx", categories) == ""

    class _Classifier:
        def classify(self, _: str) -> IntentPrediction:
            return IntentPrediction(
                label=IntentLabel.DOCUMENTATION,
                confidence=0.7,
                category_hints=[],
                tag_hints=[],
            )

    manager.set_intent_classifier(_Classifier())
    assert manager._fallback_category_from_context("Docs", categories) == "Documentation"

    manager.set_intent_classifier(None)
    assert (
        manager._fallback_category_from_context("Summary of incidents", categories) == "Reporting"
    )
    assert manager._fallback_category_from_context("misc context", []) == "General"


def test_apply_category_metadata_error_paths(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt()
    prompt.category = ""
    prompt.category_slug = ""
    assert manager._apply_category_metadata(prompt).category == ""

    class _RegistryStub:
        def ensure(self, *_: Any, **__: Any) -> PromptCategory:
            raise CategoryStorageError("db down")

    manager._category_registry = _RegistryStub()  # type: ignore[assignment]
    prompt.category = "Docs"
    with pytest.raises(PromptStorageError):
        manager._apply_category_metadata(prompt)

    class _ErrorRegistry(_RegistryStub):
        def ensure(self, *_: Any, **__: Any) -> PromptCategory:
            raise CategoryError("bad category")

    manager._category_registry = _ErrorRegistry()  # type: ignore[assignment]
    with pytest.raises(PromptManagerError):
        manager._apply_category_metadata(prompt)


def test_persist_embedding_from_worker_updates_prompt(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt()
    called: Dict[str, Any] = {}

    class _Repo:
        def update(self, prompt: Prompt) -> Prompt:
            self.last_prompt = prompt
            return prompt

    repo = _Repo()
    manager._repository = repo  # type: ignore[assignment]

    def fake_persist(prompt_arg: Prompt, embedding: Sequence[float], *, is_new: bool) -> None:
        called["prompt"] = prompt_arg
        called["embedding"] = list(embedding)
        called["is_new"] = is_new

    manager._persist_embedding = fake_persist  # type: ignore[assignment]
    manager._persist_embedding_from_worker(prompt, [0.1, 0.2])

    assert repo.last_prompt.ext4 == [0.1, 0.2]
    assert called["embedding"] == [0.1, 0.2]
    assert called["is_new"] is False


def test_apply_rating_handles_fetch_and_update_failures(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt()

    def failing_get_prompt(_: uuid.UUID) -> Prompt:
        raise PromptManagerError("missing")

    manager.get_prompt = failing_get_prompt  # type: ignore[assignment]
    manager._apply_rating(prompt.id, 4.5)

    def returning_get_prompt(_: uuid.UUID) -> Prompt:
        return prompt

    manager.get_prompt = returning_get_prompt  # type: ignore[assignment]

    def failing_update(_: Prompt) -> Prompt:
        raise PromptManagerError("db down")

    manager.update_prompt = failing_update  # type: ignore[assignment]
    manager._apply_rating(prompt.id, 5.0)


def test_clear_usage_logs_creates_and_resets(tmp_path: Path) -> None:
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "repo.db"),
        chroma_client=_DummyChromaClient(_DummyCollection()),
        enable_background_sync=False,
    )
    logs_path = tmp_path / "logs"
    manager.clear_usage_logs(logs_path)
    sample = logs_path / "entry.txt"
    sample.write_text("data")
    manager.clear_usage_logs(logs_path)
    assert logs_path.exists()
    assert not any(logs_path.iterdir())


def test_manager_close_releases_resources(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    class _Worker:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class _RedisClient:
        def __init__(self) -> None:
            self.closed = False
            self.disconnected = False
            self.connection_pool = types.SimpleNamespace(
                disconnect=lambda: setattr(self, "disconnected", True)
            )

        def close(self) -> None:
            self.closed = True

    class _Chroma:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    worker = _Worker()
    redis_client = _RedisClient()
    chroma_client = _Chroma()
    manager._embedding_worker = worker  # type: ignore[assignment]
    manager._redis_client = redis_client  # type: ignore[assignment]
    manager._chroma_client = chroma_client  # type: ignore[assignment]
    manager._closed = False

    manager.close()
    assert worker.stopped
    assert redis_client.closed
    assert redis_client.disconnected
    assert chroma_client.closed
    manager.close()  # second call should be a no-op
