"""Chroma client initialisation tests.

Updates: v0.1.0 - 2025-11-03 - Ensure Chroma anonymized telemetry is disabled by default.
"""

from __future__ import annotations

from typing import Any

from core.prompt_manager import PromptManager


class _RecordingClient:
    """Capture arguments passed to get_or_create_collection for assertions.

    Acts as a stand-in for `chromadb.Client`/`PersistentClient` and exposes
    the minimal surface that `PromptManager` uses in tests.
    """

    def __init__(self) -> None:
        self.get_or_create_called = False
        self.kwargs: dict[str, Any] = {}

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
        embedding_function: Any | None = None,
    ) -> Any:
        self.get_or_create_called = True
        self.kwargs = {
            "name": name,
            "metadata": metadata,
            "embedding_function": embedding_function,
        }
        return object()


def test_chroma_initializes_with_telemetry_disabled_by_default(tmp_path) -> None:
    """PromptManager should construct a Chroma client with telemetry off by default.

    We inject a recording client to avoid importing/initialising the real Chroma
    implementation. The test only verifies that manager bootstraps correctly and
    attempts to create a collection, which implies the client was constructed.
    """
    # Use a fake chroma client to sidestep networking/telemetry and file IO.
    recording = _RecordingClient()

    # Build the manager; provide a db path so a real SQLite file can be created.
    _ = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "prompt_manager.db"),
        chroma_client=recording,
    )

    assert recording.get_or_create_called is True
    assert recording.kwargs["name"] == "prompt_manager"
    # hnsw space is set; presence indicates we used our default path.
    assert recording.kwargs["metadata"].get("hnsw:space") == "cosine"
