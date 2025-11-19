"""Prompt Manager package façade and orchestration layer.

Updates: v0.13.11 - 2025-11-05 - Support LiteLLM workflow routing between fast and inference models.
Updates: v0.13.10 - 2025-11-05 - Add SQLite repository maintenance helpers.
Updates: v0.13.9 - 2025-11-05 - Add ChromaDB integrity verification helper.
Updates: v0.13.8 - 2025-11-05 - Add ChromaDB persistence maintenance helpers.
Updates: v0.13.7 - 2025-11-30 - Add data reset helpers for maintenance workflows.
Updates: v0.13.6 - 2025-11-26 - Track LiteLLM streaming configuration and expose runtime toggle.
Updates: v0.13.5 - 2025-11-26 - Support LiteLLM streaming execution with optional callbacks.
Updates: v0.13.4 - 2025-11-25 - Expose prompt catalogue statistics for maintenance surfaces.
Updates: v0.13.3 - 2025-11-19 - Add prompt scenario generation workflow and GUI integration.
Updates: v0.13.2 - 2025-11-16 - Add task template CRUD APIs and apply workflow.
Updates: v0.12.0 - 2025-11-15 - Add LiteLLM-backed prompt engineering workflow.
Updates: v0.11.0 - 2025-11-12 - Add multi-turn chat execution with conversation history logging.
Updates: v0.10.0 - 2025-11-11 - Emit notifications for managed LLM and embedding tasks.
Updates: v0.9.0 - 2025-11-11 - Track single-user preferences and personalise intent suggestions.
Updates: v0.8.0 - 2025-11-09 - Capture prompt ratings from executions and update quality scores.
Updates: v0.7.0 - 2025-11-08 - Integrate prompt execution pipeline with history logging.
Updates: v0.6.0 - 2025-11-06 - Add intent-aware search suggestions and ranking helpers.
Updates: v0.5.0 - 2025-11-05 - Integrate LiteLLM name generator and catalogue seeding.
Updates: v0.4.1 - 2025-11-05 - Add lifecycle shutdown hooks and mute Chroma telemetry.
Updates: v0.4.0 - 2025-11-05 - Add embedding provider with background sync and retry handling.
Updates: v0.3.0 - 2025-11-03 - Require explicit DB path; accept resolved settings inputs.
Updates: v0.2.0 - 2025-10-31 - Add SQLite repository integration with ChromaDB/Redis sync.
Updates: v0.1.0 - 2025-10-30 - Initial PromptManager with CRUD and search support.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast

import json
import logging
import os
import shutil
import sqlite3
import uuid

# ---------------------------------------------------------------------------
# NOTE: Transitional refactor
# ---------------------------
# This module is in the process of being split into a dedicated package
# `core.prompt_manager` as part of the ongoing modularisation effort
# (see AGENTS.md guidelines – KISS, DRY, maintainability).  Common exception
# classes have been migrated to ``core.exceptions`` so that they can be shared
# across sub‑modules once the split is complete.
# ---------------------------------------------------------------------------

from config import LITELLM_ROUTED_WORKFLOWS

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "none")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "0")


def _mute_posthog_capture() -> None:
    try:
        import posthog  # type: ignore
    except Exception:
        return

    def _noop_capture(*_: Any, **__: Any) -> None:
        return None

    try:
        posthog.capture = _noop_capture  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        posthog.disabled = True  # type: ignore[attr-defined]
    except Exception:
        pass


_mute_posthog_capture()


def _normalise_conversation(
    messages: Optional[Sequence[Mapping[str, str]]],
) -> List[Dict[str, str]]:
    """Return a sanitised copy of conversation messages for execution and logging."""

    normalised: List[Dict[str, str]] = []
    if not messages:
        return normalised
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        if not role:
            raise PromptExecutionError(f"Conversation entry {index} is missing a role.")
        content = message.get("content")
        if content is None:
            raise PromptExecutionError(f"Conversation entry {index} is missing content.")
        normalised.append({"role": role, "content": str(content)})
    return normalised

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis optional during development
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore[misc,assignment]

from models.prompt_model import Prompt, PromptExecution, TaskTemplate, UserProfile

from ..embedding import EmbeddingGenerationError, EmbeddingProvider, EmbeddingSyncWorker
from ..execution import CodexExecutionResult, CodexExecutor, ExecutionError
from ..history_tracker import HistoryTracker, HistoryTrackerError
from ..intent_classifier import IntentClassifier, IntentPrediction, rank_by_hints
from ..name_generation import (
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
    DescriptionGenerationError,
    NameGenerationError,
)
from ..scenario_generation import LiteLLMScenarioGenerator, ScenarioGenerationError
from ..prompt_engineering import (
    PromptEngineer,
    PromptEngineeringError,
    PromptRefinement,
)
from ..repository import (
    PromptCatalogueStats,
    PromptRepository,
    RepositoryError,
    RepositoryNotFoundError,
)
from ..notifications import (
    NotificationCenter,
    NotificationLevel,
    notification_center as default_notification_center,
)

# Re‑export shared exception classes from centralised module to preserve the
# original import path ``core.prompt_manager.PromptManagerError`` etc. during
# the deprecation window.
from ..exceptions import (  # noqa: F401 – re‑export for backward compatibility
    PromptManagerError,
    PromptNotFoundError,
    PromptExecutionUnavailable,
    PromptExecutionError,
    PromptHistoryError,
    PromptStorageError,
    PromptCacheError,
    PromptEngineeringUnavailable,
)


logger = logging.getLogger(__name__)

# Public exports of this module (until full split is finished)
__all__ = [
    # Exceptions
    "PromptManagerError",
    "PromptNotFoundError",
    "PromptExecutionUnavailable",
    "PromptExecutionError",
    "PromptHistoryError",
    "PromptStorageError",
    "PromptCacheError",
    "PromptEngineeringUnavailable",
    # Main class will be added later when moved.
]


class PromptManager:
    """Manage prompt persistence, caching, and semantic search."""

    def __init__(
        self,
        chroma_path: str,
        db_path: Union[str, Path, None] = None,
        collection_name: str = "prompt_manager",
        cache_ttl_seconds: int = 300,
        redis_client: Optional["redis.Redis"] = None,
        chroma_client: Optional[ClientAPI] = None,
        embedding_function: Optional[Any] = None,
        repository: Optional[PromptRepository] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_worker: Optional[EmbeddingSyncWorker] = None,
        enable_background_sync: bool = True,
        name_generator: Optional[LiteLLMNameGenerator] = None,
        description_generator: Optional[LiteLLMDescriptionGenerator] = None,
        scenario_generator: Optional[LiteLLMScenarioGenerator] = None,
        prompt_engineer: Optional[PromptEngineer] = None,
        fast_model: Optional[str] = None,
        inference_model: Optional[str] = None,
        workflow_models: Optional[Mapping[str, str]] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        notification_center: Optional[NotificationCenter] = None,
        executor: Optional[CodexExecutor] = None,
        user_profile: Optional[UserProfile] = None,
        history_tracker: Optional[HistoryTracker] = None,
    ) -> None:
        """Initialise the manager with data backends.

        Args:
            chroma_path: Filesystem path where ChromaDB stores data.
            db_path: Filesystem path to the SQLite database file.
            collection_name: Name of the ChromaDB collection for prompts.
            cache_ttl_seconds: Expiration for Redis cache entries.
            redis_client: Optional Redis client instance for caching.
            chroma_client: Optional preconfigured Chroma client.
            embedding_function: Optional embedding function for Chroma.
            repository: Optional preconfigured repository instance (e.g. for testing).
            fast_model: LiteLLM fast-tier model identifier configured for workflows.
            inference_model: LiteLLM inference-tier model identifier configured for workflows.
            workflow_models: Mapping of workflow identifiers to routing tiers.
            notification_center: Optional NotificationCenter override for task events.
            executor: Optional CodexExecutor responsible for running prompts.
            history_tracker: Optional execution history tracker for persistence.
            user_profile: Optional profile to seed single-user personalisation state.
            prompt_engineer: Optional prompt refinement helper using LiteLLM.
        """
        # Allow tests to supply repository directly; only require db_path when building one.
        if repository is None and db_path is None:
            raise ValueError("db_path must be provided when no repository is supplied")

        self._closed = False
        self._cache_ttl_seconds = cache_ttl_seconds
        self._redis_client = redis_client
        self._chroma_client: Any
        self._collection: Any = cast(Any, None)
        resolved_db_path = Path(db_path).expanduser() if db_path is not None else None
        resolved_chroma_path = Path(chroma_path).expanduser()
        self._db_path = resolved_db_path
        self._chroma_path = str(resolved_chroma_path)
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._logs_path = Path("data") / "logs"
        try:
            self._repository = repository or PromptRepository(str(resolved_db_path))
        except RepositoryError as exc:
            raise PromptStorageError("Unable to initialise SQLite repository") from exc
        # Disable Chroma anonymized telemetry to avoid noisy PostHog errors in some environments
        # and respect privacy defaults. Use persistent client at the configured path.
        if chroma_client is None:
            try:
                from chromadb.config import Settings as ChromaSettings

                chroma_settings = ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=str(resolved_chroma_path),
                )
                self._chroma_client = cast(Any, chromadb.Client(chroma_settings))
            except Exception:
                # Fallback to legacy PersistentClient signature if settings import or usage fails
                self._chroma_client = cast(
                    Any,
                    chromadb.PersistentClient(path=str(resolved_chroma_path)),
                )
        else:
            self._chroma_client = cast(Any, chroma_client)
        self._initialise_chroma_collection()

        self._notification_center = notification_center or default_notification_center

        self._embedding_provider = embedding_provider or EmbeddingProvider(embedding_function)
        if enable_background_sync:
            worker_logger = logger.getChild("embedding_sync")
            self._embedding_worker = embedding_worker or EmbeddingSyncWorker(
                provider=self._embedding_provider,
                fetch_prompt=self._repository.get,
                persist_callback=self._persist_embedding_from_worker,
                notification_center=self._notification_center,
                logger=worker_logger,
            )
        else:
            self._embedding_worker = embedding_worker or _NullEmbeddingWorker()
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._scenario_generator = scenario_generator
        self._prompt_engineer = prompt_engineer
        self._litellm_fast_model: Optional[str] = self._normalise_model_identifier(fast_model)
        if self._litellm_fast_model is None and getattr(name_generator, "model", None):
            self._litellm_fast_model = self._normalise_model_identifier(getattr(name_generator, "model"))
        self._litellm_inference_model: Optional[str] = self._normalise_model_identifier(inference_model)
        self._litellm_workflow_models: Dict[str, str] = {}
        if workflow_models:
            for key, value in workflow_models.items():
                key_str = str(key).strip()
                if key_str not in LITELLM_ROUTED_WORKFLOWS:
                    continue
                if value is None:
                    continue
                choice = str(value).strip().lower()
                if choice == "inference":
                    self._litellm_workflow_models[key_str] = "inference"
        self._litellm_stream: bool = False
        self._litellm_drop_params: Optional[Sequence[str]] = None
        self._litellm_reasoning_effort: Optional[str] = None
        for candidate in (name_generator, scenario_generator, prompt_engineer, executor):
            if candidate is not None and getattr(candidate, "drop_params", None):
                params = getattr(candidate, "drop_params")
                self._litellm_drop_params = tuple(params)  # type: ignore[arg-type]
                break
        for candidate in (executor, name_generator, scenario_generator, prompt_engineer):
            if candidate is not None and hasattr(candidate, "stream"):
                self._litellm_stream = bool(getattr(candidate, "stream", False))
                if self._litellm_stream:
                    break
        if executor is not None and getattr(executor, "reasoning_effort", None):
            effort = getattr(executor, "reasoning_effort")
            self._litellm_reasoning_effort = str(effort) if effort else None
        self._intent_classifier = intent_classifier or IntentClassifier()
        self._executor = executor
        if self._executor is not None:
            if self._litellm_drop_params:
                self._executor.drop_params = list(self._litellm_drop_params)
            if self._litellm_reasoning_effort:
                self._executor.reasoning_effort = self._litellm_reasoning_effort
            self._executor.stream = self._litellm_stream
        self._history_tracker = history_tracker
        if user_profile is not None:
            self._user_profile: Optional[UserProfile] = user_profile
        else:
            try:
                self._user_profile = self._repository.get_user_profile()
            except RepositoryError:
                logger.warning("Unable to load persisted user profile", exc_info=True)
                self._user_profile = None

    @staticmethod
    def _normalise_model_identifier(value: Optional[str]) -> Optional[str]:
        """Return a stripped model identifier when provided."""

        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _initialise_chroma_collection(self) -> None:
        """Create or refresh the Chroma collection backing prompt embeddings."""

        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_function,
            )
            self._collection = cast(Any, collection)
        except ChromaError as exc:
            raise PromptStorageError("Unable to initialise ChromaDB collection") from exc

    def _persist_chroma_client(self) -> None:
        """Flush the Chroma client to disk when supported."""

        persist = getattr(self._chroma_client, "persist", None)
        if not callable(persist):
            return
        try:
            persist()
        except Exception:  # noqa: BLE001
            logger.warning("Unable to flush Chroma client persistence state", exc_info=True)

    def _resolve_repository_path(self) -> Path:
        """Return the path to the SQLite repository, ensuring it exists."""

        candidate = self._db_path or getattr(self._repository, "_db_path", None)
        if candidate is None:
            raise PromptStorageError("SQLite repository path is not configured.")
        if isinstance(candidate, Path):
            path = candidate
        else:
            path = Path(str(candidate))
        path = path.expanduser()
        if not path.exists():
            raise PromptStorageError(f"SQLite repository missing at {path}.")
        return path

    @dataclass(slots=True)
    class IntentSuggestions:
        """Intent-aware search recommendations returned to callers."""

        prediction: IntentPrediction
        prompts: List[Prompt]
        fallback_used: bool = False

    @dataclass(slots=True)
    class ExecutionOutcome:
        """Aggregate data returned after executing a prompt."""

        result: CodexExecutionResult
        history_entry: Optional[PromptExecution]
        conversation: List[Dict[str, str]]

    @dataclass(slots=True)
    class TemplateApplication:
        """Result returned when applying a task template."""

        template: TaskTemplate
        prompts: List[Prompt]

    @property
    def collection(self) -> Collection:
        """Expose the underlying Chroma collection."""
        return cast(Collection, self._collection)

    @property
    def repository(self) -> PromptRepository:
        """Expose the SQLite repository."""
        return self._repository

    @property
    def db_path(self) -> Optional[Path]:
        """Return the configured SQLite database path."""

        return self._db_path

    @property
    def chroma_path(self) -> Path:
        """Return the filesystem path backing the Chroma vector store."""

        return Path(self._chroma_path)

    @property
    def logs_path(self) -> Path:
        """Return the directory where application logs are written."""

        return self._logs_path

    @property
    def intent_classifier(self) -> Optional[IntentClassifier]:
        """Expose the configured intent classifier for tooling hooks."""

        return self._intent_classifier

    @property
    def executor(self) -> Optional[CodexExecutor]:
        """Return the configured prompt executor."""

        return self._executor

    def set_executor(self, executor: Optional[CodexExecutor]) -> None:
        """Assign or replace the Codex executor at runtime."""

        self._executor = executor

    @property
    def history_tracker(self) -> Optional[HistoryTracker]:
        """Expose the execution history tracker if configured."""

        return self._history_tracker

    def set_history_tracker(self, tracker: Optional[HistoryTracker]) -> None:
        """Assign or replace the history tracker at runtime."""

        self._history_tracker = tracker

    @property
    def notification_center(self) -> NotificationCenter:
        """Expose the notification centre for UI and tooling integrations."""

        return self._notification_center

    @property
    def user_profile(self) -> Optional[UserProfile]:
        """Return the stored single-user profile when available."""

        return self._user_profile

    @property
    def prompt_engineer(self) -> Optional[PromptEngineer]:
        """Return the configured prompt engineering helper, if any."""

        return self._prompt_engineer

    def get_redis_details(self) -> Dict[str, Any]:
        """Return connection and usage details for the configured Redis cache."""

        details: Dict[str, Any] = {"enabled": self._redis_client is not None}
        client = self._redis_client
        if client is None:
            return details
        try:
            ping_ok = bool(client.ping())
        except RedisError as exc:
            details.update({"enabled": True, "status": "error", "error": str(exc)})
            return details
        details["status"] = "online" if ping_ok else "offline"

        connection: Dict[str, Any] = {}
        pool = getattr(client, "connection_pool", None)
        if pool is not None:
            kwargs = getattr(pool, "connection_kwargs", {}) or {}
            host = kwargs.get("host") or kwargs.get("unix_socket_path")
            if host:
                connection["host"] = host
            if kwargs.get("port") is not None:
                connection["port"] = kwargs.get("port")
            if kwargs.get("db") is not None:
                connection["database"] = kwargs.get("db")
            if kwargs.get("username"):
                connection["username"] = kwargs.get("username")
            if kwargs.get("ssl"):
                connection["ssl"] = bool(kwargs.get("ssl"))
        if connection:
            details["connection"] = connection

        try:
            dbsize = client.dbsize()
        except RedisError:
            dbsize = None
        if dbsize is not None:
            details.setdefault("stats", {})["keys"] = int(dbsize)

        try:
            info = client.info()
        except RedisError as exc:
            details.setdefault("stats", {})["info_error"] = str(exc)
        else:
            stats = details.setdefault("stats", {})
            for key in ("used_memory_human", "used_memory_peak_human", "maxmemory_human"):
                if info.get(key) is not None:
                    stats[key] = info[key]
            hits = info.get("keyspace_hits")
            misses = info.get("keyspace_misses")
            if hits is not None:
                stats["hits"] = hits
            if misses is not None:
                stats["misses"] = misses
        if hits and misses is not None:
            total = hits + misses
            if total:
                stats["hit_rate"] = round((hits / total) * 100, 2)
        if "role" in info:
            details["role"] = info["role"]
        return details

    def get_chroma_details(self) -> Dict[str, Any]:
        """Return filesystem and collection metrics for the configured Chroma store."""

        details: Dict[str, Any] = {"enabled": self._collection is not None}
        details["path"] = self._chroma_path
        details["collection"] = self._collection_name
        collection = self._collection
        if collection is None:
            return details
        try:
            count = collection.count()
        except ChromaError as exc:
            details["status"] = "error"
            details["error"] = str(exc)
        else:
            details["status"] = "online"
            details.setdefault("stats", {})["documents"] = count
        # Estimate on-disk size
        try:
            path_obj = Path(self._chroma_path)
            if path_obj.exists():
                size_bytes = sum(
                    entry.stat().st_size
                    for entry in path_obj.rglob("*")
                    if entry.is_file()
                )
                details.setdefault("stats", {})["disk_usage_bytes"] = size_bytes
        except (OSError, ValueError):
            pass
        return details

    def reset_prompt_repository(self) -> None:
        """Clear all prompts, templates, executions, and profiles from SQLite storage."""

        reset_func = getattr(self._repository, "reset_all_data", None)
        if not callable(reset_func):
            raise PromptManagerError("Repository reset is unavailable.")
        try:
            reset_func()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to reset prompt repository") from exc
        logger.info("Prompt repository reset completed.")
        self.refresh_user_profile()

    def reset_vector_store(self) -> None:
        """Remove all embeddings from the Chroma vector store."""

        if self._collection is None:
            return
        try:
            delete_collection = getattr(self._chroma_client, "delete_collection", None)
            if callable(delete_collection):
                delete_collection(name=self._collection_name)
                self._initialise_chroma_collection()
            else:
                self._collection.delete(where={})
        except Exception as exc:  # noqa: BLE001
            raise PromptStorageError("Unable to reset Chroma vector store") from exc
        logger.info("Chroma vector store reset completed.")

    def compact_vector_store(self) -> None:
        """Vacuum and truncate the persistent Chroma SQLite store."""

        db_path = Path(self._chroma_path) / "chroma.sqlite3"
        if not db_path.exists():
            raise PromptStorageError(f"Chroma persistence database missing at {db_path}.")
        self._persist_chroma_client()
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                connection.execute("VACUUM;")
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to compact Chroma vector store") from exc
        logger.info("Chroma vector store VACUUM completed at %s", db_path)

    def optimize_vector_store(self) -> None:
        """Refresh SQLite statistics to optimize Chroma query planning."""

        db_path = Path(self._chroma_path) / "chroma.sqlite3"
        if not db_path.exists():
            raise PromptStorageError(f"Chroma persistence database missing at {db_path}.")
        self._persist_chroma_client()
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
                connection.execute("ANALYZE;")
                try:
                    connection.execute("PRAGMA optimize;")
                except sqlite3.Error as pragma_error:
                    logger.debug(
                        "PRAGMA optimize not supported by current SQLite build: %s",
                        pragma_error,
                    )
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to optimize Chroma vector store") from exc
        logger.info("Chroma vector store optimization completed at %s", db_path)

    def verify_vector_store(self) -> str:
        """Run integrity checks against the persistent Chroma store."""

        db_path = Path(self._chroma_path) / "chroma.sqlite3"
        if not db_path.exists():
            raise PromptStorageError(f"Chroma persistence database missing at {db_path}.")

        diagnostics: List[str] = []
        self._persist_chroma_client()
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
                integrity_rows = connection.execute("PRAGMA integrity_check;").fetchall()
                integrity_failures = [str(row[0]) for row in integrity_rows if str(row[0]).lower() != "ok"]
                if integrity_failures:
                    message = "; ".join(integrity_failures)
                    raise PromptStorageError(f"Chroma integrity check failed: {message}")
                diagnostics.append("SQLite integrity_check: ok")

                quick_rows = connection.execute("PRAGMA quick_check;").fetchall()
                quick_failures = [str(row[0]) for row in quick_rows if str(row[0]).lower() != "ok"]
                if quick_failures:
                    message = "; ".join(quick_failures)
                    raise PromptStorageError(f"Chroma quick_check failed: {message}")
                diagnostics.append("SQLite quick_check: ok")
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to verify Chroma vector store") from exc

        if self._collection is None:
            self._initialise_chroma_collection()

        collection = self._collection
        if collection is None:
            raise PromptStorageError("Chroma collection could not be initialised for verification.")

        try:
            count = int(collection.count())
            diagnostics.append(f"Collection count: {count}")
            collection.peek(limit=min(count or 1, 10))
            diagnostics.append("Collection peek: ok")
        except ChromaError as exc:
            raise PromptStorageError("Unable to query Chroma collection during verification") from exc

        summary = "\n".join(diagnostics)
        logger.info("Chroma vector store verification completed successfully: %s", summary.replace("\n", " | "))
        return summary

    def compact_repository(self) -> None:
        """Vacuum the SQLite prompt repository to reclaim disk space."""

        db_path = self._resolve_repository_path()
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                connection.execute("VACUUM;")
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to compact SQLite repository") from exc
        logger.info("SQLite repository VACUUM completed at %s", db_path)

    def optimize_repository(self) -> None:
        """Refresh SQLite statistics for the prompt repository."""

        db_path = self._resolve_repository_path()
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
                connection.execute("ANALYZE;")
                try:
                    connection.execute("PRAGMA optimize;")
                except sqlite3.Error as pragma_error:
                    logger.debug(
                        "Repository PRAGMA optimize not supported by current SQLite build: %s",
                        pragma_error,
                    )
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to optimize SQLite repository") from exc
        logger.info("SQLite repository optimization completed at %s", db_path)

    def verify_repository(self) -> str:
        """Run integrity checks against the SQLite prompt repository."""

        db_path = self._resolve_repository_path()
        diagnostics: List[str] = []
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
                integrity_rows = connection.execute("PRAGMA integrity_check;").fetchall()
                integrity_failures = [str(row[0]) for row in integrity_rows if str(row[0]).lower() != "ok"]
                if integrity_failures:
                    message = "; ".join(integrity_failures)
                    raise PromptStorageError(f"SQLite integrity check failed: {message}")
                diagnostics.append("SQLite integrity_check: ok")

                quick_rows = connection.execute("PRAGMA quick_check;").fetchall()
                quick_failures = [str(row[0]) for row in quick_rows if str(row[0]).lower() != "ok"]
                if quick_failures:
                    message = "; ".join(quick_failures)
                    raise PromptStorageError(f"SQLite quick_check failed: {message}")
                diagnostics.append("SQLite quick_check: ok")

                prompt_count = connection.execute("SELECT COUNT(*) FROM prompts;").fetchone()
                template_count = connection.execute("SELECT COUNT(*) FROM templates;").fetchone()
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to verify SQLite repository") from exc

        prompts_total = int(prompt_count[0]) if prompt_count else 0
        templates_total = int(template_count[0]) if template_count else 0
        diagnostics.append(f"Prompts: {prompts_total}")
        diagnostics.append(f"Templates: {templates_total}")

        summary = "\n".join(diagnostics)
        logger.info("SQLite repository verification completed successfully: %s", summary.replace("\n", " | "))
        return summary

    def clear_usage_logs(self, logs_path: Optional[Union[str, Path]] = None) -> None:
        """Remove persisted usage analytics logs while keeping settings intact."""

        path = Path(logs_path) if logs_path is not None else self._logs_path
        path = path.expanduser()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Usage log directory created at %s", path)
            return
        try:
            for entry in path.iterdir():
                if entry.is_file() or entry.is_symlink():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
        except OSError as exc:
            raise PromptManagerError(f"Unable to clear usage logs: {exc}") from exc
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Usage logs cleared at %s", path)

    def reset_application_data(self, *, clear_logs: bool = True) -> None:
        """Reset prompt data, embeddings, and optional usage logs."""

        self.reset_prompt_repository()
        self.reset_vector_store()
        if clear_logs:
            self.clear_usage_logs()

    def get_prompt_catalogue_stats(self) -> PromptCatalogueStats:
        """Return aggregate prompt statistics for maintenance workflows."""

        try:
            return self._repository.get_prompt_catalogue_stats()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to compute prompt catalogue statistics") from exc

    def refresh_user_profile(self) -> Optional[UserProfile]:
        """Reload the persisted profile from the repository."""

        try:
            self._user_profile = self._repository.get_user_profile()
        except RepositoryError:
            logger.warning("Unable to refresh user profile from storage", exc_info=True)
        return self._user_profile

    def generate_prompt_name(self, context: str) -> str:
        """Return a prompt name using the configured LiteLLM generator."""
        if self._name_generator is None:
            raise NameGenerationError(
                "LiteLLM name generator is not configured. Set PROMPT_MANAGER_LITELLM_MODEL."
            )
        task_id = f"name-gen:{uuid.uuid4()}"
        metadata = {"context_length": len(context or "")}
        with self._notification_center.track_task(
            title="Prompt name generation",
            task_id=task_id,
            start_message="Generating prompt name via LiteLLM…",
            success_message="Prompt name generated.",
            failure_message="Prompt name generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                suggestion = self._name_generator.generate(context)
            except Exception as exc:
                raise NameGenerationError(str(exc)) from exc
        return suggestion

    def generate_prompt_description(self, context: str) -> str:
        """Return a prompt description using the configured LiteLLM generator."""

        if self._description_generator is None:
            raise DescriptionGenerationError(
                "LiteLLM description generator is not configured. Set PROMPT_MANAGER_LITELLM_MODEL."
            )
        task_id = f"description-gen:{uuid.uuid4()}"
        metadata = {"context_length": len(context or "")}
        with self._notification_center.track_task(
            title="Prompt description generation",
            task_id=task_id,
            start_message="Generating prompt description via LiteLLM…",
            success_message="Prompt description generated.",
            failure_message="Prompt description generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                summary = self._description_generator.generate(context)
            except Exception as exc:
                raise DescriptionGenerationError(str(exc)) from exc
        return summary

    def generate_prompt_scenarios(self, context: str, *, max_scenarios: int = 3) -> List[str]:
        """Return usage scenarios for a prompt via the configured LiteLLM helper."""

        if self._scenario_generator is None:
            raise ScenarioGenerationError(
                "LiteLLM scenario generator is not configured. Set PROMPT_MANAGER_LITELLM_MODEL."
            )
        task_id = f"scenario-gen:{uuid.uuid4()}"
        metadata = {
            "context_length": len(context or ""),
            "max_scenarios": max(0, int(max_scenarios)),
        }
        with self._notification_center.track_task(
            title="Prompt scenario generation",
            task_id=task_id,
            start_message="Generating scenarios via LiteLLM…",
            success_message="Prompt scenarios generated.",
            failure_message="Prompt scenario generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                scenarios = self._scenario_generator.generate(
                    context,
                    max_scenarios=max_scenarios,
                )
            except Exception as exc:
                raise ScenarioGenerationError(str(exc)) from exc
        return scenarios

    def refine_prompt_text(
        self,
        prompt_text: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        negative_constraints: Optional[Sequence[str]] = None,
    ) -> PromptRefinement:
        """Improve a prompt using the configured prompt engineer."""

        if not prompt_text.strip():
            raise PromptEngineeringError("Prompt refinement requires non-empty prompt text.")
        if self._prompt_engineer is None:
            raise PromptEngineeringUnavailable(
                "Prompt engineering is not configured. Set PROMPT_MANAGER_LITELLM_MODEL to enable refinement."
            )
        task_id = f"prompt-refine:{uuid.uuid4()}"
        metadata = {
            "prompt_length": len(prompt_text or ""),
            "has_name": bool(name),
            "tag_count": len(tags or []),
        }
        with self._notification_center.track_task(
            title="Prompt refinement",
            task_id=task_id,
            start_message="Analysing prompt via LiteLLM…",
            success_message="Prompt refined.",
            failure_message="Prompt refinement failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                return self._prompt_engineer.refine(
                    prompt_text,
                    name=name,
                    description=description,
                    category=category,
                    tags=tags,
                    negative_constraints=negative_constraints,
                )
            except PromptEngineeringError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise PromptEngineeringError("Prompt refinement failed unexpectedly.") from exc

    def execute_prompt(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
        stream: Optional[bool] = None,
        on_stream: Optional[Callable[[str], None]] = None,
    ) -> "PromptManager.ExecutionOutcome":
        """Execute a prompt via LiteLLM and persist the outcome when configured."""

        if not request_text.strip():
            raise PromptExecutionError("Prompt execution requires non-empty input text.")
        if self._executor is None:
            raise PromptExecutionUnavailable(
                "Prompt execution is not configured. Provide LiteLLM credentials and model."
            )

        conversation_history = _normalise_conversation(conversation)
        prompt = self.get_prompt(prompt_id)
        task_id = f"prompt-exec:{prompt.id}:{uuid.uuid4()}"
        metadata = {
            "prompt_id": str(prompt.id),
            "prompt_name": prompt.name,
            "request_length": len(request_text or ""),
        }
        with self._notification_center.track_task(
            title="Prompt execution",
            task_id=task_id,
            start_message=f"Running '{prompt.name}' via LiteLLM…",
            success_message=f"Completed '{prompt.name}'.",
            failure_message=f"Prompt execution failed for '{prompt.name}'",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                result = self._executor.execute(
                    prompt,
                    request_text,
                    conversation=conversation_history,
                    stream=stream,
                    on_stream=on_stream,
                )
            except ExecutionError as exc:
                failed_messages = list(conversation_history)
                failed_messages.append({"role": "user", "content": request_text.strip()})
                self._log_execution_failure(
                    prompt.id,
                    request_text,
                    str(exc),
                    conversation=failed_messages,
                )
                raise PromptExecutionError(str(exc)) from exc

            augmented_conversation = list(conversation_history)
            augmented_conversation.append({"role": "user", "content": request_text.strip()})
            if result.response_text:
                augmented_conversation.append(
                    {"role": "assistant", "content": result.response_text}
                )
            history_entry = self._log_execution_success(
                prompt.id,
                request_text,
                result,
                conversation=augmented_conversation,
            )
            try:
                self.increment_usage(prompt.id)
            except PromptManagerError:
                logger.debug(
                    "Prompt executed but usage counter update failed",
                    extra={"prompt_id": str(prompt.id)},
                    exc_info=True,
                )

        return PromptManager.ExecutionOutcome(
            result=result,
            history_entry=history_entry,
            conversation=augmented_conversation,
        )

    def save_execution_result(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        response_text: str,
        *,
        duration_ms: Optional[int] = None,
        usage: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        rating: Optional[float] = None,
    ) -> PromptExecution:
        """Persist a manual prompt execution entry (e.g., from GUI Save Result)."""

        tracker = self._history_tracker
        if tracker is None:
            raise PromptExecutionUnavailable(
                "Execution history is not configured; cannot save results manually."
            )
        payload_metadata: Dict[str, Any] = {"manual": True}
        if metadata:
            payload_metadata.update(dict(metadata))
        if usage:
            payload_metadata.setdefault("usage", dict(usage))
        if rating is not None:
            payload_metadata["rating"] = rating
        try:
            execution = tracker.record_success(
                prompt_id=prompt_id,
                request_text=request_text,
                response_text=response_text,
                duration_ms=duration_ms,
                metadata=payload_metadata,
                rating=rating,
            )
        except HistoryTrackerError as exc:
            raise PromptHistoryError(str(exc)) from exc
        if rating is not None:
            self._apply_rating(prompt_id, rating)
        return execution

    def update_execution_note(self, execution_id: uuid.UUID, note: Optional[str]) -> PromptExecution:
        """Update the note metadata for a history entry."""

        tracker = self._history_tracker
        if tracker is None:
            raise PromptExecutionUnavailable(
                "Execution history is not configured; cannot update saved results."
            )
        try:
            return tracker.update_note(execution_id, note)
        except HistoryTrackerError as exc:
            raise PromptHistoryError(str(exc)) from exc

    def list_recent_executions(self, *, limit: int = 20) -> List[PromptExecution]:
        """Return recently logged executions if history tracking is enabled."""

        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.list_recent(limit=limit)
        except HistoryTrackerError:
            logger.warning("Unable to list execution history", exc_info=True)
            return []

    def list_executions_for_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> List[PromptExecution]:
        """Return execution history for a specific prompt."""

        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.list_for_prompt(prompt_id, limit=limit)
        except HistoryTrackerError:
            logger.warning(
                "Unable to list execution history for prompt",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return []

    def query_executions(
        self,
        *,
        status: Optional[str] = None,
        prompt_id: Optional[uuid.UUID] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[PromptExecution]:
        """Return executions filtered by status, prompt, and search term."""

        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.query_executions(
                status=status,
                prompt_id=prompt_id,
                search=search,
                limit=limit,
            )
        except HistoryTrackerError:
            logger.warning("Unable to query execution history", exc_info=True)
            return []

    def set_name_generator(
        self,
        model: Optional[str],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        *,
        inference_model: Optional[str] = None,
        workflow_models: Optional[Mapping[str, str]] = None,
        drop_params: Optional[Sequence[str]] = None,
        reasoning_effort: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> None:
        """Configure LiteLLM-backed workflows at runtime."""

        self._litellm_fast_model = self._normalise_model_identifier(model)
        self._litellm_inference_model = self._normalise_model_identifier(inference_model)
        routing: Dict[str, str] = {}
        if workflow_models:
            for key, value in workflow_models.items():
                workflow_key = str(key).strip()
                if workflow_key not in LITELLM_ROUTED_WORKFLOWS:
                    continue
                if value is None:
                    continue
                choice = str(value).strip().lower()
                if choice == "inference":
                    routing[workflow_key] = "inference"
        self._litellm_workflow_models = routing

        if drop_params is not None:
            cleaned_params = [str(item).strip() for item in drop_params if str(item).strip()]
            self._litellm_drop_params = tuple(cleaned_params) if cleaned_params else None
        else:
            self._litellm_drop_params = None

        self._litellm_reasoning_effort = reasoning_effort
        if stream is not None:
            self._litellm_stream = bool(stream)

        # Reset existing helpers before rebuilding.
        self._name_generator = None
        self._description_generator = None
        self._prompt_engineer = None
        self._scenario_generator = None
        self._executor = None

        if not (self._litellm_fast_model or self._litellm_inference_model):
            self._litellm_reasoning_effort = None
            self._litellm_stream = False
            return

        drop_params_payload = (
            list(self._litellm_drop_params) if self._litellm_drop_params else None
        )

        def _select_model(workflow: str) -> Optional[str]:
            selection = self._litellm_workflow_models.get(workflow, "fast")
            if selection == "inference":
                return self._litellm_inference_model or self._litellm_fast_model
            return self._litellm_fast_model or self._litellm_inference_model

        def _construct(factory: Callable[..., Any], workflow: str, **extra: Any) -> Optional[Any]:
            selected_model = _select_model(workflow)
            if not selected_model:
                return None
            return factory(
                model=selected_model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                drop_params=drop_params_payload,
                **extra,
            )

        try:
            self._name_generator = _construct(LiteLLMNameGenerator, "name_generation")
            self._description_generator = _construct(
                LiteLLMDescriptionGenerator, "description_generation"
            )
            self._prompt_engineer = _construct(PromptEngineer, "prompt_engineering")
            self._scenario_generator = _construct(
                LiteLLMScenarioGenerator, "scenario_generation"
            )
            self._executor = _construct(
                CodexExecutor,
                "prompt_execution",
                reasoning_effort=self._litellm_reasoning_effort,
                stream=self._litellm_stream,
            )
        except RuntimeError as exc:
            raise NameGenerationError(str(exc)) from exc

        if self._executor is not None:
            if self._litellm_drop_params:
                self._executor.drop_params = list(self._litellm_drop_params)
            if self._litellm_reasoning_effort:
                self._executor.reasoning_effort = self._litellm_reasoning_effort
            self._executor.stream = self._litellm_stream

        if self._intent_classifier is not None and (
            self._litellm_fast_model or self._litellm_inference_model
        ):
            logger.debug("LiteLLM powered features enabled for intent classifier")

    def _log_execution_success(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        result: CodexExecutionResult,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> Optional[PromptExecution]:
        """Persist a successful execution outcome when the tracker is available."""

        tracker = self._history_tracker
        if tracker is None:
            return None
        usage_metadata: Dict[str, Any] = {}
        try:
            usage_metadata = dict(result.usage)
        except Exception:  # pragma: no cover - defensive
            usage_metadata = {}
        metadata: Dict[str, Any] = {"usage": usage_metadata}
        if conversation:
            metadata["conversation"] = list(conversation)
        try:
            return tracker.record_success(
                prompt_id=prompt_id,
                request_text=request_text,
                response_text=result.response_text,
                duration_ms=result.duration_ms,
                metadata=metadata,
            )
        except HistoryTrackerError:
            logger.warning(
                "Prompt executed but history logging failed",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return None

    def _log_execution_failure(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        error_message: str,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> Optional[PromptExecution]:
        """Persist a failed execution attempt when history tracking is enabled."""

        tracker = self._history_tracker
        if tracker is None:
            return None
        metadata: Optional[Mapping[str, Any]] = None
        if conversation:
            metadata = {"conversation": list(conversation)}
        try:
            return tracker.record_failure(
                prompt_id=prompt_id,
                request_text=request_text,
                error_message=error_message,
                metadata=metadata,
            )
        except HistoryTrackerError:
            logger.warning(
                "Prompt execution failed and could not be logged",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return None

    def set_intent_classifier(self, classifier: Optional[IntentClassifier]) -> None:
        """Replace the runtime intent classifier implementation."""

        self._intent_classifier = classifier

    def close(self) -> None:
        """Release background workers and backend connections."""
        if self._closed:
            return

        self._closed = True

        worker = getattr(self, "_embedding_worker", None)
        stop = getattr(worker, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                logger.debug("Failed to stop embedding worker cleanly", exc_info=True)

        if self._redis_client is not None:
            redis_close = getattr(self._redis_client, "close", None)
            if callable(redis_close):
                try:
                    redis_close()
                except Exception:
                    logger.debug(
                        "Failed to close Redis client cleanly",
                        exc_info=True,
                    )
            pool = getattr(self._redis_client, "connection_pool", None)
            disconnect = getattr(pool, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    logger.debug(
                        "Failed to disconnect Redis connection pool cleanly",
                        exc_info=True,
                    )

        chroma_client = getattr(self, "_chroma_client", None)
        if chroma_client is not None:
            close_fn = getattr(chroma_client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.debug(
                        "Failed to close Chroma client cleanly",
                        exc_info=True,
                    )

    def __enter__(self) -> "PromptManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - dynamic signature
        self.close()

    def create_prompt(
        self,
        prompt: Prompt,
        embedding: Optional[Sequence[float]] = None,
    ) -> Prompt:
        """Persist a new prompt in SQLite/ChromaDB and prime the cache."""
        generated_embedding: Optional[List[float]] = None
        if embedding is not None:
            generated_embedding = list(embedding)
        else:
            try:
                generated_embedding = self._embedding_provider.embed(prompt.document)
            except EmbeddingGenerationError:
                logger.warning(
                    "Falling back to background embedding for prompt",
                    extra={"prompt_id": str(prompt.id)},
                )
        try:
            if generated_embedding is not None:
                prompt.ext4 = list(generated_embedding)
            stored_prompt = self._repository.add(prompt)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist prompt {prompt.id}") from exc
        if generated_embedding is not None:
            try:
                self._persist_embedding(stored_prompt, generated_embedding, is_new=True)
            except PromptStorageError as exc:
                try:
                    self._repository.delete(prompt.id)
                except RepositoryError:
                    logger.error(
                        "Unable to roll back SQLite insert after Chroma failure",
                        extra={"prompt_id": str(prompt.id)},
                    )
                raise exc
        else:
            self._embedding_worker.schedule(stored_prompt.id)
            try:
                self._cache_prompt(stored_prompt)
            except PromptCacheError:
                logger.warning(
                    "Prompt created but not cached",
                    extra={"prompt_id": str(prompt.id)},
                )
        return stored_prompt

    def get_prompt(self, prompt_id: uuid.UUID) -> Prompt:
        """Retrieve a prompt from cache or SQLite."""
        prompt = self._get_cached_prompt(prompt_id)
        if prompt:
            return prompt
        try:
            prompt_obj = self._repository.get(prompt_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt_id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to fetch prompt {prompt_id} from SQLite") from exc
        try:
            self._cache_prompt(prompt_obj)
        except PromptCacheError:
            logger.debug(
                "Fetched prompt could not be cached",
                extra={"prompt_id": str(prompt_id)},
            )
        return prompt_obj

    def update_prompt(
        self,
        prompt: Prompt,
        embedding: Optional[Sequence[float]] = None,
    ) -> Prompt:
        """Update an existing prompt with new metadata."""
        generated_embedding: Optional[List[float]] = None
        if embedding is not None:
            generated_embedding = list(embedding)
        else:
            try:
                generated_embedding = self._embedding_provider.embed(prompt.document)
            except EmbeddingGenerationError:
                logger.warning(
                    "Scheduling background embedding refresh",
                    extra={"prompt_id": str(prompt.id)},
                )
        try:
            if generated_embedding is not None:
                prompt.ext4 = list(generated_embedding)
            updated_prompt = self._repository.update(prompt)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt.id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to update prompt {prompt.id} in SQLite") from exc
        if generated_embedding is not None:
            self._persist_embedding(updated_prompt, generated_embedding, is_new=False)
        else:
            self._embedding_worker.schedule(updated_prompt.id)
            try:
                self._cache_prompt(updated_prompt)
            except PromptCacheError:
                logger.warning(
                    "Prompt updated but cache refresh failed",
                    extra={"prompt_id": str(prompt.id)},
                )
        return updated_prompt

    def delete_prompt(self, prompt_id: uuid.UUID) -> None:
        """Remove a prompt from all data stores."""
        try:
            self._repository.delete(prompt_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt_id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to delete prompt {prompt_id} from SQLite") from exc
        try:
            self._collection.delete(ids=[str(prompt_id)])
        except ChromaError as exc:
            raise PromptStorageError(f"Failed to delete prompt {prompt_id}") from exc
        try:
            self._evict_cached_prompt(prompt_id)
        except PromptCacheError:
            logger.debug(
                "Prompt deleted but cache eviction failed",
                extra={"prompt_id": str(prompt_id)},
            )

    # Template management ------------------------------------------------ #

    def list_templates(self, *, include_inactive: bool = False) -> List[TaskTemplate]:
        """Return available task templates."""

        try:
            return self._repository.list_templates(include_inactive=include_inactive)
        except RepositoryError as exc:
            raise PromptStorageError("Unable to list task templates") from exc

    def get_template(self, template_id: uuid.UUID) -> TaskTemplate:
        """Return a single task template."""

        try:
            return self._repository.get_template(template_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Unable to load template {template_id}") from exc

    def create_template(self, template: TaskTemplate) -> TaskTemplate:
        """Persist a new task template."""

        template.touch()
        try:
            stored = self._repository.add_template(template)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist template {template.id}") from exc
        return stored

    def update_template(self, template: TaskTemplate) -> TaskTemplate:
        """Persist updates to an existing task template."""

        template.touch()
        try:
            updated = self._repository.update_template(template)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to update template {template.id}") from exc
        return updated

    def delete_template(self, template_id: uuid.UUID) -> None:
        """Delete a task template."""

        try:
            self._repository.delete_template(template_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to delete template {template_id}") from exc

    def apply_template(self, template_id: uuid.UUID) -> "PromptManager.TemplateApplication":
        """Return template metadata and associated prompts for task workflows."""

        template = self.get_template(template_id)
        try:
            prompts = self._repository.get_prompts_for_ids(template.prompt_ids)
        except RepositoryError as exc:
            raise PromptStorageError("Unable to load prompts for template") from exc
        missing = len(template.prompt_ids) - len(prompts)
        if missing > 0:
            logger.warning(
                "Template references missing prompts",
                extra={
                    "template_id": str(template.id),
                    "expected": len(template.prompt_ids),
                    "resolved": len(prompts),
                },
            )
        return PromptManager.TemplateApplication(template=template, prompts=prompts)

    def search_prompts(
        self,
        query_text: str,
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[Prompt]:
        """Search prompts semantically using a text query or embedding."""
        if not query_text and embedding is None:
            raise ValueError("query_text or embedding must be provided")

        query_embedding: Optional[List[float]]
        if embedding is not None:
            query_embedding = list(embedding)
        else:
            try:
                query_embedding = self._embedding_provider.embed(query_text)
            except EmbeddingGenerationError as exc:
                raise PromptStorageError("Failed to generate query embedding") from exc

        try:
            results = cast(
                Dict[str, Any],
                self._collection.query(
                    query_texts=None,
                    query_embeddings=[query_embedding] if query_embedding is not None else None,
                    n_results=limit,
                    where=where,
                    # Request fields compatible with older Chroma versions –
                    # "ids" are always returned so we omit it to avoid a
                    # validation error when the client lists supported names.
                    include=["documents", "metadatas", "distances"],
                ),
            )
        except ChromaError as exc:
            raise PromptStorageError("Failed to query prompts") from exc

        prompts: List[Prompt] = []
        ids = cast(List[str], results.get("ids", [[]])[0])
        documents = cast(List[str], results.get("documents", [[]])[0])
        metadatas = cast(List[Dict[str, Any]], results.get("metadatas", [[]])[0])
        distances = cast(List[float], results.get("distances", [[]])[0] if "distances" in results else [None] * len(ids))

        for prompt_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            try:
                prompt_uuid = uuid.UUID(prompt_id)
            except ValueError:
                logger.warning(
                    "Invalid prompt UUID in Chroma results",
                    extra={"prompt_id": prompt_id},
                )
                continue
            try:
                prompt_record = self._repository.get(prompt_uuid)
            except RepositoryNotFoundError:
                record = {"id": prompt_id, "document": document, "metadata": metadata}
                prompt_record = Prompt.from_chroma(record)
            except RepositoryError as exc:
                raise PromptStorageError(
                    f"Failed to hydrate prompt {prompt_id} from SQLite"
                ) from exc
            # Attach similarity score (1 - cosine distance) when distance is provided
            try:
                if distance is not None:
                    similarity = 1.0 - float(distance)
                    setattr(prompt_record, "_similarity", similarity)
            except Exception:  # pragma: no cover – defensive
                pass

            prompts.append(prompt_record)

        # Ensure prompts are ordered by descending similarity if available
        if any(hasattr(p, "_similarity") for p in prompts):
            prompts.sort(key=lambda p: getattr(p, "_similarity", 0.0), reverse=True)

        return prompts

    def suggest_prompts(self, query_text: str, *, limit: int = 5) -> "PromptManager.IntentSuggestions":
        """Return intent-ranked prompt recommendations for the supplied query."""

        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        stripped = query_text.strip()
        if not stripped:
            try:
                baseline = self.repository.list(limit=limit)
            except RepositoryError as exc:
                raise PromptStorageError("Unable to load prompts for suggestions") from exc
            personalised = self._personalize_ranked_prompts(baseline)
            return PromptManager.IntentSuggestions(
                IntentPrediction.general(), personalised[:limit], fallback_used=True
            )

        prediction = (
            self._intent_classifier.classify(stripped)
            if self._intent_classifier is not None
            else IntentPrediction.general()
        )

        augmented_query_parts = [stripped]
        if prediction.category_hints:
            augmented_query_parts.append(
                "Intent categories: " + ", ".join(prediction.category_hints)
            )
        if prediction.tag_hints:
            augmented_query_parts.append("Intent tags: " + ", ".join(prediction.tag_hints))
        augmented_query = "\n".join(augmented_query_parts)

        suggestions: List[Prompt] = []
        fallback_used = False
        try:
            raw_results = self.search_prompts(augmented_query, limit=max(limit * 2, 10))
        except PromptManagerError:
            raw_results = []

        ranked = rank_by_hints(
            raw_results,
            category_hints=prediction.category_hints,
            tag_hints=prediction.tag_hints,
        )
        ranked = self._personalize_ranked_prompts(ranked)

        seen_ids: set[uuid.UUID] = set()
        for prompt in ranked:
            if prompt.id in seen_ids:
                continue
            suggestions.append(prompt)
            seen_ids.add(prompt.id)
            if len(suggestions) >= limit:
                break

        if len(suggestions) < limit:
            fallback_used = True
            try:
                fallback_results = self.search_prompts(stripped, limit=max(limit * 2, 10))
            except PromptManagerError:
                fallback_results = []
            else:
                fallback_results = self._personalize_ranked_prompts(fallback_results)
            for prompt in fallback_results:
                if prompt.id in seen_ids:
                    continue
                suggestions.append(prompt)
                seen_ids.add(prompt.id)
                if len(suggestions) >= limit:
                    break

        if not suggestions:
            fallback_used = True
            try:
                suggestions = self.repository.list(limit=limit)
            except RepositoryError as exc:
                raise PromptStorageError("Unable to load prompts for suggestions") from exc

        personalised = self._personalize_ranked_prompts(suggestions)
        return PromptManager.IntentSuggestions(
            prediction=prediction,
            prompts=personalised[:limit],
            fallback_used=fallback_used,
        )

    def _personalize_ranked_prompts(self, prompts: Sequence[Prompt]) -> List[Prompt]:
        """Bias prompt order using stored user preferences while preserving stability."""

        if not prompts:
            return []
        profile = self._user_profile
        if profile is None:
            return list(prompts)

        favorite_categories = profile.favorite_categories(limit=5)
        favorite_tags = profile.favorite_tags(limit=8)
        if not favorite_categories and not favorite_tags:
            return list(prompts)

        category_weights = {name: (len(favorite_categories) - idx) * 2 for idx, name in enumerate(favorite_categories)}
        tag_weights = {name: len(favorite_tags) - idx for idx, name in enumerate(favorite_tags)}

        scored: List[tuple[float, int, Prompt]] = []
        for index, prompt in enumerate(prompts):
            score = 0.0
            category = (prompt.category or "").strip()
            if category in category_weights:
                score += float(category_weights[category])
            for tag in prompt.tags or []:
                weight = tag_weights.get(tag)
                if weight:
                    score += float(weight)
            scored.append((score, index, prompt))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [prompt for _, _, prompt in scored]

    def increment_usage(self, prompt_id: uuid.UUID) -> None:
        """Increment usage counter for a prompt."""
        prompt = self.get_prompt(prompt_id)
        prompt.usage_count += 1
        self.update_prompt(prompt)
        self._record_prompt_usage(prompt)

    def _record_prompt_usage(self, prompt: Prompt) -> None:
        """Persist prompt usage into the single-user profile when possible."""

        try:
            profile = self._repository.record_user_prompt_usage(prompt)
        except RepositoryError:
            logger.debug(
                "Failed to update user profile preferences",
                extra={"prompt_id": str(prompt.id)},
                exc_info=True,
            )
            return
        self._user_profile = profile

    def _apply_rating(self, prompt_id: uuid.UUID, rating: float) -> None:
        """Update prompt aggregates from a new rating."""
        try:
            prompt = self.get_prompt(prompt_id)
        except PromptManagerError:
            logger.warning(
                "Unable to fetch prompt for rating update",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return

        prompt.rating_count += 1
        prompt.rating_sum += float(rating)
        if prompt.rating_count > 0:
            prompt.quality_score = round(prompt.rating_sum / prompt.rating_count, 2)
        prompt.last_modified = datetime.now(timezone.utc)

        try:
            self.update_prompt(prompt)
        except PromptManagerError:
            logger.warning(
                "Unable to persist rating update for prompt",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )

    # Cache helpers ----------------------------------------------------- #

    def _cache_prompt(self, prompt: Prompt) -> None:
        """Store prompt representation in Redis."""
        if self._redis_client is None:
            return
        payload = json.dumps(prompt.to_record(), ensure_ascii=False)
        try:
            self._redis_client.setex(
                self._cache_key(prompt.id),
                self._cache_ttl_seconds,
                payload,
            )
        except RedisError as exc:  # pragma: no cover - redis not in CI
            raise PromptCacheError("Failed to write prompt to Redis") from exc

    def _get_cached_prompt(self, prompt_id: uuid.UUID) -> Optional[Prompt]:
        """Fetch prompt from Redis cache when available."""
        if self._redis_client is None:
            return None
        try:
            cached = self._redis_client.get(self._cache_key(prompt_id))
        except RedisError as exc:  # pragma: no cover - redis not in CI
            raise PromptCacheError("Failed to read prompt from Redis") from exc
        if not cached:
            return None
        cached_text = cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
        try:
            record = json.loads(cached_text)
        except json.JSONDecodeError as exc:
            logger.warning("Cannot decode cached prompt", extra={"prompt_id": str(prompt_id)})
            raise PromptCacheError("Invalid JSON cached value") from exc
        return Prompt.from_record(record)

    def _evict_cached_prompt(self, prompt_id: uuid.UUID) -> None:
        """Remove cached prompt from Redis."""
        if self._redis_client is None:
            return
        try:
            self._redis_client.delete(self._cache_key(prompt_id))
        except RedisError as exc:  # pragma: no cover - redis not in CI
            raise PromptCacheError("Failed to evict prompt from Redis") from exc

    @staticmethod
    def _cache_key(prompt_id: uuid.UUID) -> str:
        """Format cache key for prompt entries."""
        return f"prompt:{prompt_id}"

    def _persist_embedding(self, prompt: Prompt, embedding: Sequence[float], *, is_new: bool) -> None:
        """Persist embeddings to Chroma and refresh caches."""

        payload: Dict[str, Any] = {
            "ids": [str(prompt.id)],
            "documents": [prompt.document],
            "metadatas": [prompt.to_metadata()],
            "embeddings": [list(embedding)],
        }
        try:
            self._collection.upsert(**payload)
        except ChromaError as exc:
            raise PromptStorageError(f"Failed to persist embedding for prompt {prompt.id}") from exc
        try:
            self._cache_prompt(prompt)
        except PromptCacheError:
            logger.warning(
                "Prompt cached embedding refresh failed",
                extra={"prompt_id": str(prompt.id)},
            )

    def _persist_embedding_from_worker(self, prompt: Prompt, embedding: Sequence[float]) -> None:
        """Callback invoked by background worker once embedding is generated."""

        prompt.ext4 = list(embedding)
        try:
            self._repository.update(prompt)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist embedding for prompt {prompt.id}") from exc
        self._persist_embedding(prompt, embedding, is_new=False)


class _NullEmbeddingWorker:
    """Embedding worker placeholder used when background sync is disabled."""

    def schedule(self, _: uuid.UUID) -> None:  # pragma: no cover - trivial noop
        return

    def stop(self) -> None:  # pragma: no cover - trivial noop
        return


__all__ = [
    "PromptManager",
    "PromptManagerError",
    "PromptNotFoundError",
    "PromptStorageError",
    "PromptCacheError",
    "PromptExecutionUnavailable",
    "PromptExecutionError",
    "PromptEngineeringUnavailable",
    # New storage façade
    "PromptStorage",
    # Execution facade
    "PromptExecutor",
    "ExecutionResult",
    # Engineering facade
    "PromptEngineerFacade",
    "PromptEngineeringError",
    "PromptRefinement",
]

# Re‑export storage façade for external use
from .storage import PromptStorage  # noqa: E402  (after __all__)

# Execution facade re‑exports
from .execution import (  # noqa: E402
    PromptExecutor,
    ExecutionResult,
)

# Engineering facade re‑exports
from .engineering import (  # noqa: E402
    PromptEngineerFacade,
    PromptEngineeringError,
    PromptRefinement,
)
