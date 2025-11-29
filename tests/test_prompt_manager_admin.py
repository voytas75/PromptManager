"""Administrative and diagnostic tests for PromptManager."""

from __future__ import annotations

import json
import sqlite3
import types
import uuid
import zipfile
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import pytest
from chromadb.errors import ChromaError

from core.embedding import EmbeddingGenerationError
from core.exceptions import CategoryError, CategoryNotFoundError, CategoryStorageError
from core.execution import CodexExecutionResult, ExecutionError
from core.history_tracker import HistoryTrackerError
from core.intent_classifier import IntentLabel, IntentPrediction
from core.prompt_engineering import PromptEngineeringError, PromptRefinement
from core.prompt_manager import (
    CategorySuggestionError,
    DescriptionGenerationError,
    NameGenerationError,
    PromptEngineeringUnavailable,
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptManager,
    PromptManagerError,
    PromptStorageError,
    RedisError,
    ScenarioGenerationError,
)
from core.repository import PromptRepository, RepositoryError, RepositoryNotFoundError
from models.category_model import PromptCategory, slugify_category
from models.prompt_model import ExecutionStatus, Prompt, PromptExecution, PromptVersion
from prompt_templates import DEFAULT_PROMPT_TEMPLATES

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from pathlib import Path


def _clone_prompt(prompt: Prompt) -> Prompt:
    return Prompt.from_record(prompt.to_record())


def _clone_category(category: PromptCategory) -> PromptCategory:
    return PromptCategory.from_record(category.to_record())


class _InMemoryRepository:
    """Minimal in-memory repository facade for PromptManager tests."""

    def __init__(self, prompts: Optional[Sequence[Prompt]] = None) -> None:
        self.prompts: Dict[uuid.UUID, Prompt] = {}
        for prompt in prompts or []:
            self.add(prompt)
        default_category = PromptCategory(
            slug="general",
            label="General",
            description="General prompts",
        )
        self.categories: Dict[str, PromptCategory] = {default_category.slug: default_category}
        self.latest_versions: Dict[uuid.UUID, PromptVersion] = {}
        self.reset_called = False
        self.not_found_ids: set[uuid.UUID] = set()
        self.raise_on_update = False
        self.raise_on_create_category = False
        self.raise_on_update_category = False
        self.raise_list_error = False

    def list(self, limit: Optional[int] = None) -> List[Prompt]:
        if self.raise_list_error:
            raise RepositoryError("list failed")
        values = [_clone_prompt(prompt) for prompt in self.prompts.values()]
        return values if limit is None else values[:limit]

    def add(self, prompt: Prompt) -> Prompt:
        self.prompts[prompt.id] = _clone_prompt(prompt)
        return prompt

    def update(self, prompt: Prompt) -> Prompt:
        if prompt.id not in self.prompts or prompt.id in self.not_found_ids:
            raise RepositoryNotFoundError("missing prompt")
        if self.raise_on_update:
            raise RepositoryError("update failed")
        self.prompts[prompt.id] = _clone_prompt(prompt)
        return prompt

    def delete(self, prompt_id: uuid.UUID) -> None:
        if prompt_id not in self.prompts:
            raise RepositoryNotFoundError("missing prompt")
        del self.prompts[prompt_id]

    def get(self, prompt_id: uuid.UUID) -> Prompt:
        if prompt_id not in self.prompts:
            raise RepositoryNotFoundError("missing prompt")
        return _clone_prompt(self.prompts[prompt_id])

    def create_category(self, category: PromptCategory) -> PromptCategory:
        if self.raise_on_create_category:
            raise RepositoryError("create category failed")
        stored = _clone_category(category)
        self.categories[stored.slug] = stored
        return _clone_category(stored)

    def update_category(self, category: PromptCategory) -> PromptCategory:
        if category.slug not in self.categories:
            raise RepositoryNotFoundError("missing category")
        if self.raise_on_update_category:
            raise RepositoryError("update category failed")
        self.categories[category.slug] = _clone_category(category)
        return _clone_category(category)

    def set_category_active(self, slug: str, is_active: bool) -> PromptCategory:
        normalized = slugify_category(slug)
        if normalized not in self.categories:
            raise RepositoryNotFoundError("missing category")
        category = _clone_category(self.categories[normalized])
        category.is_active = is_active
        self.categories[normalized] = category
        return _clone_category(category)

    def list_categories(self, include_archived: bool = False) -> List[PromptCategory]:
        categories = list(self.categories.values())
        if not include_archived:
            categories = [cat for cat in categories if cat.is_active]
        return [_clone_category(cat) for cat in categories]

    def get_prompt_latest_version(self, prompt_id: uuid.UUID) -> Optional[PromptVersion]:
        return self.latest_versions.get(prompt_id)

    def reset_all_data(self) -> None:
        self.prompts.clear()
        self.reset_called = True


class _CategoryRegistryStub:
    """Stub category registry exposing only the surface used in tests."""

    def __init__(self, category: Optional[PromptCategory] = None) -> None:
        self.category = category
        self.refresh_calls = 0

    def refresh(self) -> List[PromptCategory]:
        self.refresh_calls += 1
        return []

    def require(self, slug: str) -> PromptCategory:
        if self.category is None:
            raise CategoryNotFoundError(f"{slug} missing")
        return _clone_category(self.category)

    def get(self, slug: Optional[str]) -> Optional[PromptCategory]:
        if slug and self.category and slugify_category(slug) == self.category.slug:
            return _clone_category(self.category)
        return None


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
def prompt_manager(
    tmp_path: "Path",
) -> tuple[PromptManager, _DummyCollection, _DummyChromaClient, "Path"]:
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


def _ensure_chroma_database(chroma_dir: "Path") -> Path:
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


def test_get_redis_details_disabled_when_client_missing(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    manager._redis_client = None  # type: ignore[assignment]
    assert manager.get_redis_details() == {"enabled": False}


def test_get_redis_details_handles_redis_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    redis_client = _RedisClientStub(
        ping_result=True,
        dbsize_value=42,
        info_error=RedisError("info fail"),
    )
    failing_dbsize = _RedisClientStub(ping_result=True, dbsize_value=42)

    def _fail_dbsize() -> int:
        raise RedisError("dbsize")

    failing_dbsize.dbsize = _fail_dbsize  # type: ignore[assignment]
    manager._redis_client = failing_dbsize  # type: ignore[assignment]
    details = manager.get_redis_details()
    assert "stats" not in details
    manager._redis_client = redis_client  # type: ignore[assignment]
    details = manager.get_redis_details()
    assert "info_error" in details["stats"]


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


def test_get_chroma_details_returns_when_collection_missing(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    manager._collection = None  # type: ignore[assignment]
    details = manager.get_chroma_details()
    assert "status" not in details


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


def test_verify_vector_store_missing_database_raises(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    db_path = chroma_dir / "chroma.sqlite3"
    if db_path.exists():
        db_path.unlink()
    with pytest.raises(PromptStorageError):
        manager.verify_vector_store()


def test_verify_vector_store_integrity_failure(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    _ensure_chroma_database(chroma_dir)

    class _IntegrityFailConnection:
        def __enter__(self) -> "_IntegrityFailConnection":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def execute(self, sql: str, *args: object) -> "_IntegrityFailConnection":
            self._last_sql = sql.lower()
            return self

        def fetchall(self) -> List[tuple[str]]:
            if "integrity_check" in getattr(self, "_last_sql", ""):
                return [("not ok",)]
            return [("ok",)]

    monkeypatch.setattr(
        "core.prompt_manager.sqlite3.connect",
        lambda *args, **kwargs: _IntegrityFailConnection(),
    )
    with pytest.raises(PromptStorageError):
        manager.verify_vector_store()


def test_verify_vector_store_quick_check_failure(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    _ensure_chroma_database(chroma_dir)

    class _QuickFailConnection:
        def __enter__(self) -> "_QuickFailConnection":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def execute(self, sql: str, *args: object) -> "_QuickFailConnection":
            self._last_sql = sql.lower()
            return self

        def fetchall(self) -> List[tuple[str]]:
            if "quick_check" in getattr(self, "_last_sql", ""):
                return [("not ok",)]
            return [("ok",)]

    monkeypatch.setattr(
        "core.prompt_manager.sqlite3.connect",
        lambda *args, **kwargs: _QuickFailConnection(),
    )
    with pytest.raises(PromptStorageError):
        manager.verify_vector_store()


def test_verify_vector_store_handles_collection_initialisation_failure(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    _ensure_chroma_database(chroma_dir)
    manager._collection = None  # type: ignore[assignment]

    def fake_initialise(self: PromptManager) -> None:
        self._collection = None

    manager._initialise_chroma_collection = fake_initialise.__get__(manager, PromptManager)
    with pytest.raises(PromptStorageError):
        manager.verify_vector_store()


def test_verify_vector_store_handles_chroma_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    _ensure_chroma_database(chroma_dir)
    failing_collection = _DummyCollection()
    failing_collection.count_exception = ChromaError("count fail")
    manager._collection = failing_collection  # type: ignore[assignment]
    with pytest.raises(PromptStorageError):
        manager.verify_vector_store()


def test_optimize_vector_store_handles_optimize_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    _ensure_chroma_database(chroma_dir)
    real_connect = sqlite3.connect

    class _ConnectionWrapper:
        def __init__(self, path: str, timeout: float = 60.0) -> None:
            self._conn = real_connect(path)

        def __enter__(self) -> "_ConnectionWrapper":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            self._conn.close()

        def execute(self, sql: str, *params: object) -> sqlite3.Cursor:
            if "PRAGMA optimize" in sql:
                raise sqlite3.OperationalError("optimize unsupported")
            return self._conn.execute(sql, *params)

    def fake_connect(path: str, timeout: float = 60.0) -> _ConnectionWrapper:
        return _ConnectionWrapper(path, timeout)

    monkeypatch.setattr("core.prompt_manager.sqlite3.connect", fake_connect)
    manager.optimize_vector_store()


def test_optimize_vector_store_missing_database_raises(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    db_path = chroma_dir / "chroma.sqlite3"
    if db_path.exists():
        db_path.unlink()
    with pytest.raises(PromptStorageError):
        manager.optimize_vector_store()


def test_optimize_vector_store_sqlite_failure(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    _ensure_chroma_database(chroma_dir)

    def fake_connect(*_: Any, **__: Any) -> None:
        raise sqlite3.OperationalError("failure")

    monkeypatch.setattr("core.prompt_manager.sqlite3.connect", fake_connect)
    with pytest.raises(PromptStorageError):
        manager.optimize_vector_store()


def test_compact_repository_handles_sqlite_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _, _ = prompt_manager

    def fake_connect(*_: Any, **__: Any) -> None:
        raise sqlite3.OperationalError("fail")

    monkeypatch.setattr("core.prompt_manager.sqlite3.connect", fake_connect)
    with pytest.raises(PromptStorageError):
        manager.compact_repository()


def test_optimize_repository_handles_sqlite_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, _, _ = prompt_manager

    def fake_connect(*_: Any, **__: Any) -> None:
        raise sqlite3.OperationalError("fail")

    monkeypatch.setattr("core.prompt_manager.sqlite3.connect", fake_connect)
    with pytest.raises(PromptStorageError):
        manager.optimize_repository()


def test_query_executions_handles_filters(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    tracker = _HistoryTrackerStub()
    manager.set_history_tracker(tracker)

    first = manager.query_executions(status="unknown", search="logs", limit=5)
    manager.query_executions(status=ExecutionStatus.SUCCESS)
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
    assert manager._fallback_category_from_context(
        "Investigate bug",
        categories,
    ) == "Reasoning / Debugging"

    manager.set_intent_classifier(None)
    assert manager._fallback_category_from_context(
        "Document the feature",
        categories,
    ) == "Documentation"


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


def test_snapshot_archive_includes_sqlite_and_chroma(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
    tmp_path: "Path",
) -> None:
    manager, _, _, chroma_dir = prompt_manager
    manager.create_prompt(_make_prompt("Snapshot prompt"))
    extra_dir = chroma_dir / "collections"
    extra_dir.mkdir(parents=True, exist_ok=True)
    (extra_dir / "dummy.bin").write_text("payload")

    archive_path = manager.create_data_snapshot(tmp_path / "snapshots")
    assert archive_path.exists()

    db_name = manager._resolve_repository_path().name  # noqa: SLF001 - test uses private helper
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert f"sqlite/{db_name}" in names
        assert "manifest.json" in names
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["sqlite"]["size_bytes"] > 0
    assert manifest["chroma"]["path"]
    stats = manifest["prompt_stats"]
    assert stats is None or stats["total_prompts"] >= 1


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


def test_refresh_prompt_scenarios_updates_prompt(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    prompt = manager.create_prompt(_make_prompt("Scenario Refresh"))

    class _ScenarioGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, context: str, *, max_scenarios: int) -> List[str]:
            self.calls += 1
            base = self.calls * 10
            return [f"Scenario {base + index}" for index in range(1, max_scenarios + 1)]

    manager._scenario_generator = _ScenarioGenerator()

    updated = manager.refresh_prompt_scenarios(prompt.id, max_scenarios=2)
    assert updated.scenarios == ["Scenario 11", "Scenario 12"]
    persisted = manager.get_prompt(prompt.id)
    assert persisted.scenarios == ["Scenario 11", "Scenario 12"]

    updated_again = manager.refresh_prompt_scenarios(prompt.id, max_scenarios=1)
    assert updated_again.scenarios == ["Scenario 21"]
    persisted_again = manager.get_prompt(prompt.id)
    assert persisted_again.scenarios == ["Scenario 21"]


def test_get_category_health_returns_counts_and_stats(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt("Health Prompt")
    prompt.category = "Documentation"
    prompt.category_slug = slugify_category(prompt.category)
    stored = manager.create_prompt(prompt)

    execution = PromptExecution(
        id=uuid.uuid4(),
        prompt_id=stored.id,
        request_text="demo",
        response_text="ok",
    )
    manager.repository.add_execution(execution)

    health_entries = manager.get_category_health()
    target = next((entry for entry in health_entries if entry.slug == prompt.category_slug), None)
    assert target is not None
    assert target.total_prompts >= 1
    assert target.active_prompts >= 1


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


def test_rebuild_embeddings_handles_reset_and_missing_prompts(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt_ok = _make_prompt("Persisted")
    prompt_missing = _make_prompt("Missing")
    repo = _InMemoryRepository([prompt_ok, prompt_missing])
    repo.not_found_ids = {prompt_missing.id}
    manager._repository = repo  # type: ignore[assignment]

    class _EmbeddingStub:
        def embed(self, text: str) -> List[float]:
            return [float(len(text)), 0.0]

    persist_calls: List[tuple[uuid.UUID, bool]] = []

    def fake_persist(prompt: Prompt, vector: Sequence[float], *, is_new: bool) -> None:
        persist_calls.append((prompt.id, is_new))

    reset_called: List[bool] = []

    def fake_reset(self: PromptManager) -> None:
        reset_called.append(True)

    manager.reset_vector_store = fake_reset.__get__(manager, PromptManager)
    manager._embedding_provider = _EmbeddingStub()  # type: ignore[assignment]
    manager._persist_embedding = fake_persist  # type: ignore[assignment]

    successes, failures = manager.rebuild_embeddings(reset_store=True)

    assert successes == 1
    assert failures == 1
    assert persist_calls == [(prompt_ok.id, True)]
    assert reset_called == [True]


def test_rebuild_embeddings_raises_when_list_fails(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    repo = _InMemoryRepository()
    repo.raise_list_error = True
    manager._repository = repo  # type: ignore[assignment]
    with pytest.raises(PromptStorageError):
        manager.rebuild_embeddings()


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


def test_log_execution_success_and_failure_paths(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt()
    tracker_calls: Dict[str, List[Dict[str, Any]]] = {"success": [], "failure": []}

    class _Tracker:
        def __init__(self) -> None:
            self.raise_on_success = False
            self.raise_on_failure = False

        def record_success(self, **kwargs: Any) -> PromptExecution:
            if self.raise_on_success:
                raise HistoryTrackerError("success fail")
            tracker_calls["success"].append(kwargs)
            return PromptExecution(
                id=uuid.uuid4(),
                prompt_id=prompt.id,
                request_text=kwargs["request_text"],
                response_text=kwargs["response_text"],
            )

        def record_failure(self, **kwargs: Any) -> PromptExecution:
            if self.raise_on_failure:
                raise HistoryTrackerError("failure fail")
            tracker_calls["failure"].append(kwargs)
            return PromptExecution(
                id=uuid.uuid4(),
                prompt_id=prompt.id,
                request_text=kwargs["request_text"],
                response_text=None,
                status=ExecutionStatus.FAILED,
            )

    tracker = _Tracker()
    manager._history_tracker = tracker  # type: ignore[assignment]
    result = CodexExecutionResult(
        prompt_id=prompt.id,
        request_text="hello",
        response_text="world",
        duration_ms=10,
        usage={"prompt_tokens": 5},
        raw_response={"choices": []},
    )
    logged_success = manager._log_execution_success(
        prompt.id,
        "hello",
        result,
        conversation=[{"role": "user", "content": "hi"}],
        context_metadata={"foo": "bar"},
    )
    assert logged_success is not None
    assert tracker_calls["success"][0]["metadata"]["usage"]["prompt_tokens"] == 5

    tracker.raise_on_success = True
    assert (
        manager._log_execution_success(prompt.id, "hello", result) is None
    ), "should swallow tracker errors"

    logged_failure = manager._log_execution_failure(
        prompt.id,
        "hello",
        "boom",
        conversation=[{"role": "assistant", "content": "oops"}],
        context_metadata={"foo": "bar"},
    )
    assert logged_failure is not None
    tracker.raise_on_failure = True
    assert manager._log_execution_failure(prompt.id, "hello", "boom") is None


def test_manager_initialises_litellm_defaults_from_helpers(tmp_path: "Path") -> None:
    class _GeneratorStub:
        def __init__(
            self,
            *,
            stream: bool = False,
            drop_params: Optional[Sequence[str]] = None,
        ) -> None:
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


def test_create_category_propagates_repository_error(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    repo = _InMemoryRepository()
    repo.raise_on_create_category = True
    manager._repository = repo  # type: ignore[assignment]
    manager._category_registry = _CategoryRegistryStub()  # type: ignore[assignment]
    with pytest.raises(CategoryStorageError):
        manager.create_category(label="Docs")


def test_update_category_refreshes_registry_and_handles_errors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    repo = _InMemoryRepository()
    manager._repository = repo  # type: ignore[assignment]
    base_category = PromptCategory(slug="docs", label="Docs", description="Docs")
    repo.categories[base_category.slug] = base_category
    registry_stub = _CategoryRegistryStub(base_category)
    manager._category_registry = registry_stub  # type: ignore[assignment]
    updated = manager.update_category("docs", label="Docs Updated")
    assert updated.label == "Docs Updated"
    assert registry_stub.refresh_calls == 1

    repo.categories.pop("docs", None)
    with pytest.raises(CategoryNotFoundError):
        manager.update_category("docs", label="Missing")

    repo.categories["docs"] = base_category
    repo.raise_on_update_category = True
    with pytest.raises(CategoryStorageError):
        manager.update_category("docs", label="Error")


def test_set_category_active_handles_missing_entries(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    repo = _InMemoryRepository()
    manager._repository = repo  # type: ignore[assignment]
    manager._category_registry = _CategoryRegistryStub()  # type: ignore[assignment]
    base_category = PromptCategory(slug="docs", label="Docs", description="Docs")
    repo.categories[base_category.slug] = base_category
    updated = manager.set_category_active("docs", False)
    assert updated.is_active is False

    with pytest.raises(CategoryNotFoundError):
        manager.set_category_active("missing", True)
def test_refine_prompt_structure_validation(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    with pytest.raises(PromptEngineeringError):
        manager.refine_prompt_structure("")
    manager._prompt_structure_engineer = None  # type: ignore[assignment]
    with pytest.raises(PromptEngineeringUnavailable):
        manager.refine_prompt_structure("Refine me")


def test_refine_prompt_structure_uses_engineer(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    class _Engineer:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def refine_structure(self, text: str, **_: Any) -> PromptRefinement:
            self.calls.append(text)
            return PromptRefinement(
                analysis="ok",
                improved_prompt=text.upper(),
                checklist=[],
                warnings=[],
                confidence=0.9,
            )

    engineer = _Engineer()
    manager._prompt_structure_engineer = engineer  # type: ignore[assignment]
    refinement = manager.refine_prompt_structure("Structure me", tags=["a"])
    assert refinement.improved_prompt == "STRUCTURE ME"
    assert engineer.calls == ["Structure me"]


def test_execute_prompt_validation(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt("Execute prompt")
    repo = _InMemoryRepository([prompt])
    manager._repository = repo  # type: ignore[assignment]
    manager._executor = None  # type: ignore[assignment]
    with pytest.raises(PromptExecutionError):
        manager.execute_prompt(prompt.id, "   ")
    with pytest.raises(PromptExecutionUnavailable):
        manager.execute_prompt(prompt.id, "Run diagnostics")


def test_execute_prompt_success_and_failure_logging(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    prompt = _make_prompt("Executable")
    repo = _InMemoryRepository([prompt])
    manager._repository = repo  # type: ignore[assignment]

    class _Executor:
        def __init__(self) -> None:
            self.stream = False
            self.model = "test-model"
            self.raise_error = False

        def execute(self, *_: Any, **__: Any) -> CodexExecutionResult:
            if self.raise_error:
                raise ExecutionError("executor failure")
            return CodexExecutionResult(
                prompt_id=prompt.id,
                request_text="hello",
                response_text="world",
                duration_ms=1,
                usage={},
                raw_response={},
            )

    executor = _Executor()
    manager._executor = executor  # type: ignore[assignment]
    manager._history_tracker = None  # type: ignore[assignment]
    manager._log_execution_success = lambda *args, **kwargs: None  # type: ignore[assignment]
    manager._log_execution_failure = lambda *args, **kwargs: None  # type: ignore[assignment]
    manager.increment_usage = lambda *_: None  # type: ignore[assignment]
    outcome = manager.execute_prompt(
        prompt.id,
        "Run task",
        conversation=[{"role": "user", "content": "hi"}],
    )
    assert outcome.result.response_text == "world"

    executor.raise_error = True
    with pytest.raises(PromptExecutionError):
        manager.execute_prompt(prompt.id, "Run task")


def test_create_prompt_embeds_and_persists(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    repo = _InMemoryRepository()
    manager._repository = repo  # type: ignore[assignment]
    prompt = _make_prompt("New Prompt")
    manager._embedding_provider = types.SimpleNamespace(embed=lambda *_: [0.1, 0.2])  # type: ignore[assignment]
    persisted: List[uuid.UUID] = []

    def fake_persist(prompt_obj: Prompt, vector: Sequence[float], *, is_new: bool) -> None:
        assert is_new is True
        persisted.append(prompt_obj.id)

    manager._persist_embedding = fake_persist  # type: ignore[assignment]
    manager._commit_prompt_version = (  # type: ignore[assignment]
        lambda *args, **kwargs: types.SimpleNamespace(id=1, version_number=1)
    )
    created = manager.create_prompt(prompt)
    assert created.id == prompt.id
    assert persisted == [prompt.id]


def test_create_prompt_schedules_worker_on_embedding_failure(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    repo = _InMemoryRepository()
    manager._repository = repo  # type: ignore[assignment]
    prompt = _make_prompt("Needs Embedding")

    class _FailingEmbeddingProvider:
        def embed(self, _: str) -> List[float]:
            raise EmbeddingGenerationError("fail")

    scheduled: List[uuid.UUID] = []
    cached: List[uuid.UUID] = []
    manager._embedding_provider = _FailingEmbeddingProvider()  # type: ignore[assignment]
    manager._embedding_worker = types.SimpleNamespace(schedule=scheduled.append)  # type: ignore[assignment]
    manager._cache_prompt = lambda prompt_obj: cached.append(prompt_obj.id)  # type: ignore[assignment]
    manager._commit_prompt_version = (  # type: ignore[assignment]
        lambda *args, **kwargs: types.SimpleNamespace(id=1, version_number=1)
    )
    manager.create_prompt(prompt)
    assert scheduled == [prompt.id]
    assert cached == [prompt.id]


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


def test_clear_usage_logs_creates_and_resets(tmp_path: "Path") -> None:
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


def test_diagnose_embeddings_reports_backend_and_store_health(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, collection, _, _ = prompt_manager
    prompt = _make_prompt("Vector ok")
    prompt.ext4 = [0.1] * 32
    manager.repository.add(prompt)
    collection.count_value = 1

    report = manager.diagnose_embeddings(sample_text="diagnostics sample")

    assert report.backend_ok is True
    assert report.chroma_ok is True
    assert report.prompts_with_embeddings == 1
    assert report.repository_total == 1
    assert report.mismatched_prompts == []
    assert report.consistent_counts is True


def test_diagnose_embeddings_flags_mismatches_and_missing_vectors(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, collection, _, _ = prompt_manager
    good_prompt = _make_prompt("Aligned")
    good_prompt.ext4 = [0.2] * 32
    manager.repository.add(good_prompt)

    bad_prompt = _make_prompt("Mismatched")
    bad_prompt.ext4 = [0.3] * 8
    manager.repository.add(bad_prompt)

    missing_prompt = _make_prompt("Missing")
    manager.repository.add(missing_prompt)

    collection.count_value = 1

    report = manager.diagnose_embeddings()

    assert len(report.mismatched_prompts) == 1
    assert report.mismatched_prompts[0].prompt_id == bad_prompt.id
    assert len(report.missing_prompts) == 1
    assert report.missing_prompts[0].prompt_id == missing_prompt.id
    assert report.consistent_counts is False


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


def test_manager_close_handles_component_exceptions(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    class _Worker:
        def stop(self) -> None:
            raise RuntimeError("stop failed")

    class _RedisClient:
        def __init__(self) -> None:
            self.connection_pool = types.SimpleNamespace(
                disconnect=self._disconnect,
            )

        def close(self) -> None:
            raise RuntimeError("close failed")

        def _disconnect(self) -> None:
            raise RuntimeError("disconnect failed")

    class _Chroma:
        def close(self) -> None:
            raise RuntimeError("close failed")

    manager._closed = False
    manager._embedding_worker = _Worker()  # type: ignore[assignment]
    manager._redis_client = _RedisClient()  # type: ignore[assignment]
    manager._chroma_client = _Chroma()  # type: ignore[assignment]
    manager.close()


def test_get_prompt_catalogue_stats_error(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager

    class _Repo:
        def get_prompt_catalogue_stats(self) -> Any:
            raise RepositoryError("stats down")

    manager._repository = _Repo()  # type: ignore[assignment]
    with pytest.raises(PromptStorageError):
        manager.get_prompt_catalogue_stats()


def test_resolve_category_label_prefers_registry(
    prompt_manager: tuple[PromptManager, _DummyCollection, _DummyChromaClient, Path],
) -> None:
    manager, _, _, _ = prompt_manager
    category = PromptCategory(slug="docs", label="Docs", description="Docs")
    stub = _CategoryRegistryStub(category)
    manager._category_registry = stub  # type: ignore[assignment]
    assert manager.resolve_category_label("docs", fallback="fallback") == "Docs"
    manager._category_registry = _CategoryRegistryStub()  # type: ignore[assignment]
    assert manager.resolve_category_label("missing", fallback="fallback") == "fallback"
