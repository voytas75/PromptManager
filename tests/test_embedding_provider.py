"""Unit tests covering embedding provider and sync worker utilities."""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from core.embedding import (
    EmbeddingGenerationError,
    EmbeddingProvider,
    EmbeddingSyncWorker,
    LiteLLMEmbeddingFunction,
    SentenceTransformersEmbeddingFunction,
    create_embedding_function,
)
from models.prompt_model import Prompt


def _make_prompt(name: str = "Embedding Test") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Sample description",
        category="tests",
    )


def test_embedding_provider_returns_deterministic_vector() -> None:
    provider = EmbeddingProvider()
    vector_one = provider.embed("Hello world")
    vector_two = provider.embed("Hello world")
    assert len(vector_one) == 32
    assert vector_one == vector_two


def test_embedding_provider_retries_before_success() -> None:
    calls = {"count": 0}

    def _flaky(texts):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return [[float(len(texts[0]))]]

    provider = EmbeddingProvider(_flaky, max_retries=2, retry_delay_seconds=0.0)
    vector = provider.embed("abcde")
    assert vector == [5.0]
    assert calls["count"] == 2

    failing_provider = EmbeddingProvider(
        lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        max_retries=1,
        retry_delay_seconds=0.0,
    )
    with pytest.raises(EmbeddingGenerationError):
        failing_provider.embed("fails")


def test_embedding_sync_worker_eventually_persists_vector() -> None:
    prompt = _make_prompt()
    persisted: list[tuple[uuid.UUID, list[float]]] = []
    completion = threading.Event()

    class _FlakyProvider:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            if self.calls == 1:
                raise EmbeddingGenerationError("retry please")
            return [0.1, 0.2, float(len(text))]

    provider = _FlakyProvider()

    def _fetch_prompt(prompt_id: uuid.UUID) -> Prompt:
        assert prompt_id == prompt.id
        return prompt

    def _persist_callback(prompt_obj: Prompt, vector: list[float]) -> None:
        persisted.append((prompt_obj.id, list(vector)))
        completion.set()

    worker = EmbeddingSyncWorker(
        provider=provider,
        fetch_prompt=_fetch_prompt,
        persist_callback=_persist_callback,
        max_attempts=3,
        retry_delay_seconds=0.01,
    )
    worker.schedule(prompt.id)

    assert completion.wait(timeout=2.0), "Background worker did not persist embedding"
    worker.stop()

    assert provider.calls >= 2
    assert persisted and persisted[0][0] == prompt.id


def test_embedding_sync_worker_stops_after_max_attempts() -> None:
    prompt = _make_prompt("Max Attempts")
    calls = {"count": 0}
    completion = threading.Event()

    def _failing_embed(_: str) -> list[float]:
        calls["count"] += 1
        raise EmbeddingGenerationError("nope")

    def _fetch_prompt(prompt_id: uuid.UUID) -> Prompt:
        if prompt_id != prompt.id:
            raise KeyError(prompt_id)
        return prompt

    def _persist(_: Prompt, __: list[float]) -> None:
        completion.set()

    worker = EmbeddingSyncWorker(
        provider=EmbeddingProvider(_failing_embed, max_retries=1, retry_delay_seconds=0.0),
        fetch_prompt=_fetch_prompt,
        persist_callback=_persist,
        max_attempts=2,
        retry_delay_seconds=0.01,
    )
    worker.schedule(prompt.id)
    time.sleep(0.2)
    worker.stop()

    assert calls["count"] == 2
    assert not completion.is_set()


def test_create_embedding_function_returns_none_for_default() -> None:
    result = create_embedding_function(
        "deterministic",
        model=None,
        api_key=None,
        api_base=None,
    )
    assert result is None


def test_create_embedding_function_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        create_embedding_function("unknown-backend", model=None, api_key=None, api_base=None)


def test_litellm_embedding_function_uses_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_get_embedding():
        class _FakeException(Exception):
            pass

        def _fake_embedding(**kwargs):
            captured["request"] = kwargs
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        return _fake_embedding, _FakeException

    monkeypatch.setattr("core.embedding.get_embedding", _fake_get_embedding)
    func = create_embedding_function(
        "litellm",
        model="text-embedding-3-small",
        api_key="secret",
        api_base="https://api.invalid",
    )
    assert isinstance(func, LiteLLMEmbeddingFunction)
    vectors = func(["sample text"])
    assert vectors == [[0.1, 0.2, 0.3]]
    assert captured["request"]["api_key"] == "secret"
    assert captured["request"]["api_base"] == "https://api.invalid"
    assert captured["request"]["model"] == "text-embedding-3-small"


def test_sentence_transformers_embedding_function_encodes(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeModel:
        def encode(self, inputs, convert_to_numpy: bool, normalize_embeddings: bool):
            assert convert_to_numpy is True
            assert normalize_embeddings is False
            return [[float(len(text))] for text in inputs]

    monkeypatch.setattr(
        SentenceTransformersEmbeddingFunction,
        "_load_model",
        lambda self: _FakeModel(),
        raising=False,
    )
    func = create_embedding_function(
        "sentence-transformers",
        model="all-MiniLM-L6-v2",
        api_key=None,
        api_base=None,
    )
    assert isinstance(func, SentenceTransformersEmbeddingFunction)
    vectors = func(["hello", "world!"])
    assert vectors == [[5.0], [6.0]]


def test_litellm_embedding_function_accepts_object_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DataEntry:
        def __init__(self) -> None:
            self.embedding = [0.4, 0.5]

    class _ModelResponse:
        def __init__(self) -> None:
            self.data = [_DataEntry()]

    def _fake_get_embedding():
        class _FakeException(Exception):
            pass

        def _fake_embedding(**_: object) -> object:
            return _ModelResponse()

        return _fake_embedding, _FakeException

    monkeypatch.setattr("core.embedding.get_embedding", _fake_get_embedding)
    func = create_embedding_function("litellm", model="test", api_key=None, api_base=None)
    assert func(["input"]) == [[0.4, 0.5]]


def test_litellm_embedding_function_uses_model_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DumpOnly:
        def model_dump(self) -> dict[str, object]:
            return {"data": [{"embedding": [0.9, 1.1]}]}

    def _fake_get_embedding():
        class _FakeException(Exception):
            pass

        def _fake_embedding(**_: object) -> object:
            return _DumpOnly()

        return _fake_embedding, _FakeException

    monkeypatch.setattr("core.embedding.get_embedding", _fake_get_embedding)
    func = create_embedding_function("litellm", model="test", api_key=None, api_base=None)
    assert func(["payload"]) == [[0.9, 1.1]]
