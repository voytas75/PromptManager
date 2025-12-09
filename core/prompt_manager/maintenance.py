"""Maintenance mixin providing operational utilities for Prompt Manager.

Updates:
  v0.1.2 - 2025-12-08 - Close SQLite handles via explicit context manager to prevent leaks.
  v0.1.1 - 2025-12-07 - Gate CollectionProtocol import behind TYPE_CHECKING.
  v0.1.0 - 2025-12-02 - Extract maintenance helpers from core.prompt_manager.__init__.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from sqlite3 import Connection as SQLiteConnection
from typing import TYPE_CHECKING, Any, Iterator, cast

from chromadb.errors import ChromaError

from ..embedding import EmbeddingGenerationError
from ..exceptions import PromptManagerError, PromptStorageError
from ..repository import (
    PromptCatalogueStats,
    RepositoryError,
    RepositoryNotFoundError,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Mapping

    from . import PromptManager as _PromptManager
    from .backends import CollectionProtocol
else:
    _PromptManager = Any
    CollectionProtocol = Any

logger = logging.getLogger(__name__)


@contextmanager
def _sqlite_connection(path: Path, *, timeout: float = 60.0) -> Iterator[SQLiteConnection]:
    """Yield a SQLite connection that commits changes and closes reliably."""
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        yield conn
        commit = getattr(conn, "commit", None)
        if callable(commit):
            commit()
    except Exception:
        rollback = getattr(conn, "rollback", None)
        if callable(rollback):
            rollback()
        raise
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            close()

__all__ = ["MaintenanceMixin"]

_parent_module = sys.modules.get("core.prompt_manager")
if _parent_module is not None:
    sqlite3 = getattr(_parent_module, "sqlite3", sqlite3)


def _coerce_int(value: Any) -> int | None:
    """Best-effort conversion of Redis statistics into integers."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


class MaintenanceMixin:
"""Operational helpers for repository, cache, and vector-store maintenance.

Updates:
  v0.1.1 - 2025-12-09 - Include Redis unavailability reasons in maintenance details.
  v0.1.0 - 2025-12-03 - Initial maintenance helpers.
"""

    _redis_client: Any | None
    _collection: CollectionProtocol | None
    _collection_name: str
    _chroma_path: str
    _chroma_client: Any
    _repository: Any
    _embedding_provider: Any
    _logs_path: Path

    def _as_prompt_manager(self) -> _PromptManager:
        """Return self casted to PromptManager for cross-mixin helpers."""
        return cast("_PromptManager", self)

    def get_redis_details(self) -> dict[str, Any]:
        """Return connection and usage details for the configured Redis cache."""
        manager = self._as_prompt_manager()
        details: dict[str, Any] = {"enabled": manager._redis_client is not None}
        client = manager._redis_client
        if client is None:
            reason = getattr(manager, "redis_unavailable_reason", None)
            if reason:
                details["reason"] = reason
                details["status"] = "disabled"
            return details

        try:
            ping_ok = bool(client.ping())
        except Exception as exc:  # pragma: no cover - Redis not exercised in CI
            details.update({"enabled": True, "status": "error", "error": str(exc)})
            return details

        details["status"] = "online" if ping_ok else "offline"

        connection: dict[str, Any] = {}
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

        stats: dict[str, Any] = {}
        hits: int | None = None
        misses: int | None = None

        try:
            dbsize = client.dbsize()
        except Exception:  # pragma: no cover - Redis not exercised in CI
            dbsize = None
        if dbsize is not None:
            stats["keys"] = int(dbsize)

        info_data: Mapping[str, Any] | None = None
        try:
            info_data = client.info()
        except Exception as exc:  # pragma: no cover - Redis not exercised in CI
            stats["info_error"] = str(exc)
        else:
            for key in ("used_memory_human", "used_memory_peak_human", "maxmemory_human"):
                value = info_data.get(key)
                if value is not None:
                    stats[key] = value
            hits = _coerce_int(info_data.get("keyspace_hits"))
            misses = _coerce_int(info_data.get("keyspace_misses"))
            if hits is not None:
                stats["hits"] = hits
            if misses is not None:
                stats["misses"] = misses
            role = info_data.get("role")
            if role is not None:
                details["role"] = role

        if hits is not None and misses is not None:
            total = hits + misses
            if total:
                stats["hit_rate"] = round((hits / total) * 100, 2)

        if stats:
            details["stats"] = stats
        return details

    def get_chroma_details(self) -> dict[str, Any]:
        """Return filesystem and collection metrics for the configured Chroma store."""
        manager = self._as_prompt_manager()
        details: dict[str, Any] = {"enabled": manager._collection is not None}
        details["path"] = manager._chroma_path
        details["collection"] = manager._collection_name
        collection = manager._collection
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
        try:
            path_obj = Path(manager._chroma_path)
            if path_obj.exists():
                size_bytes = sum(
                    entry.stat().st_size for entry in path_obj.rglob("*") if entry.is_file()
                )
                details.setdefault("stats", {})["disk_usage_bytes"] = size_bytes
        except (OSError, ValueError):  # pragma: no cover - filesystem specific
            pass
        return details

    def reset_prompt_repository(self) -> None:
        """Clear all prompts, executions, and profiles from SQLite storage."""
        manager = self._as_prompt_manager()
        reset_func = getattr(manager._repository, "reset_all_data", None)
        if not callable(reset_func):
            raise PromptManagerError("Repository reset is unavailable.")
        try:
            reset_func()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to reset prompt repository") from exc
        logger.info("Prompt repository reset completed.")
        manager.refresh_user_profile()

    def reset_vector_store(self) -> None:
        """Remove all embeddings from the Chroma vector store."""
        manager = self._as_prompt_manager()
        if manager._collection is None:
            return
        try:
            delete_collection = getattr(manager._chroma_client, "delete_collection", None)
            if callable(delete_collection):
                delete_collection(name=manager._collection_name)
                manager._initialise_chroma_collection()
            else:
                manager._collection.delete(where={})
        except Exception as exc:  # pragma: no cover - backend specific
            raise PromptStorageError("Unable to reset Chroma vector store") from exc
        logger.info("Chroma vector store reset completed.")

    def rebuild_embeddings(self, *, reset_store: bool = False) -> tuple[int, int]:
        """Regenerate embeddings for all prompts and persist them to Chroma."""
        manager = self._as_prompt_manager()
        if reset_store:
            manager.reset_vector_store()

        try:
            prompts = manager._repository.list()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to load prompts for embedding rebuild") from exc

        if not prompts:
            logger.info("No prompts available for embedding rebuild.")
            return 0, 0

        successes = 0
        failures = 0
        for prompt in prompts:
            try:
                vector = manager._embedding_provider.embed(prompt.document)
            except EmbeddingGenerationError as exc:
                failures += 1
                logger.warning(
                    "Embedding generation failed during rebuild",
                    extra={"prompt_id": str(prompt.id)},
                    exc_info=exc,
                )
                continue

            prompt.ext4 = list(vector)
            try:
                manager._repository.update(prompt)
            except RepositoryNotFoundError:
                failures += 1
                logger.warning(
                    "Prompt disappeared during rebuild",
                    extra={"prompt_id": str(prompt.id)},
                )
                continue
            except RepositoryError as exc:
                raise PromptStorageError(
                    f"Failed to persist regenerated embedding for prompt {prompt.id}"
                ) from exc

            manager._persist_embedding(prompt, vector, is_new=reset_store)
            successes += 1

        logger.info(
            "Embedding rebuild finished: %s succeeded, %s failed.",
            successes,
            failures,
        )
        return successes, failures

    def compact_vector_store(self) -> None:
        """Vacuum and truncate the persistent Chroma SQLite store."""
        manager = self._as_prompt_manager()
        db_path = Path(manager._chroma_path) / "chroma.sqlite3"
        if not db_path.exists():
            raise PromptStorageError(f"Chroma persistence database missing at {db_path}.")
        manager._persist_chroma_client()
        try:
            with _sqlite_connection(db_path) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                connection.execute("VACUUM;")
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to compact Chroma vector store") from exc
        logger.info("Chroma vector store VACUUM completed at %s", db_path)

    def optimize_vector_store(self) -> None:
        """Refresh SQLite statistics to optimize Chroma query planning."""
        manager = self._as_prompt_manager()
        db_path = Path(manager._chroma_path) / "chroma.sqlite3"
        if not db_path.exists():
            raise PromptStorageError(f"Chroma persistence database missing at {db_path}.")
        manager._persist_chroma_client()
        try:
            with _sqlite_connection(db_path) as connection:
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
        manager = self._as_prompt_manager()
        db_path = Path(manager._chroma_path) / "chroma.sqlite3"
        if not db_path.exists():
            raise PromptStorageError(f"Chroma persistence database missing at {db_path}.")

        try:
            collection = manager.collection
        except PromptManagerError:
            manager._initialise_chroma_collection()
            try:
                collection = manager.collection
            except PromptManagerError as exc:
                raise PromptStorageError("Unable to initialise Chroma collection") from exc

        try:
            document_count = int(collection.count())
        except ChromaError as exc:  # pragma: no cover - backend specific
            raise PromptStorageError("Unable to query Chroma collection") from exc
        try:
            collection.peek(limit=document_count or 1)
        except ChromaError as exc:  # pragma: no cover - backend specific
            raise PromptStorageError("Unable to inspect Chroma collection") from exc

        diagnostics: list[str] = []
        manager._persist_chroma_client()
        try:
            with _sqlite_connection(db_path) as connection:
                integrity_rows = connection.execute("PRAGMA integrity_check;").fetchall()
                integrity_failures = [
                    str(row[0]) for row in integrity_rows if str(row[0]).lower() != "ok"
                ]
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
        diagnostics.append(f"Collection count: {document_count}")
        summary = "\n".join(diagnostics)
        logger.info(
            "Chroma vector store verification completed successfully: %s",
            summary.replace("\n", " | "),
        )
        return summary

    def compact_repository(self) -> None:
        """Vacuum the SQLite prompt repository to reclaim disk space."""
        manager = self._as_prompt_manager()
        db_path = manager._resolve_repository_path()
        try:
            with _sqlite_connection(db_path) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                connection.execute("VACUUM;")
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to compact SQLite repository") from exc
        logger.info("SQLite repository VACUUM completed at %s", db_path)

    def optimize_repository(self) -> None:
        """Refresh SQLite statistics for the prompt repository."""
        manager = self._as_prompt_manager()
        db_path = manager._resolve_repository_path()
        try:
            with _sqlite_connection(db_path) as connection:
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
        manager = self._as_prompt_manager()
        db_path = manager._resolve_repository_path()
        diagnostics: list[str] = []
        try:
            with _sqlite_connection(db_path) as connection:
                integrity_rows = connection.execute("PRAGMA integrity_check;").fetchall()
                integrity_failures = [
                    str(row[0]) for row in integrity_rows if str(row[0]).lower() != "ok"
                ]
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
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to verify SQLite repository") from exc

        prompts_total = int(prompt_count[0]) if prompt_count else 0
        diagnostics.append(f"Prompts: {prompts_total}")

        summary = "\n".join(diagnostics)
        logger.info(
            "SQLite repository verification completed successfully: %s",
            summary.replace("\n", " | "),
        )
        return summary

    def create_data_snapshot(self, destination: str | Path) -> Path:
        """Zip the SQLite repository, Chroma store, and a manifest for backups."""
        manager = self._as_prompt_manager()
        db_path = manager._resolve_repository_path()
        chroma_path = Path(manager._chroma_path).expanduser()
        manager._persist_chroma_client()

        timestamp_label = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        target = Path(destination).expanduser()
        if target.exists() and target.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            archive_path = target / f"prompt-manager-snapshot-{timestamp_label}.zip"
        else:
            archive_path = target if target.suffix.lower() == ".zip" else target.with_suffix(".zip")
            archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path = archive_path.resolve()

        manifest = self._build_snapshot_manifest(db_path, chroma_path)

        try:
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(db_path, arcname=f"sqlite/{db_path.name}")
                manager._write_chroma_directory(archive, chroma_path)
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                )
        except OSError as exc:
            raise PromptManagerError(f"Unable to create snapshot archive: {exc}") from exc

        logger.info("Snapshot archive created at %s", archive_path)
        return archive_path

    def clear_usage_logs(self, logs_path: str | Path | None = None) -> None:
        """Remove persisted usage analytics logs while keeping settings intact."""
        manager = self._as_prompt_manager()
        path = Path(logs_path) if logs_path is not None else manager._logs_path
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
        manager = self._as_prompt_manager()
        manager.reset_prompt_repository()
        manager.reset_vector_store()
        if clear_logs:
            manager.clear_usage_logs()

    def get_prompt_catalogue_stats(self) -> PromptCatalogueStats:
        """Return aggregate prompt statistics for maintenance workflows."""
        manager = self._as_prompt_manager()
        try:
            return manager._repository.get_prompt_catalogue_stats()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to compute prompt catalogue statistics") from exc

    def _build_snapshot_manifest(self, db_path: Path, chroma_path: Path) -> dict[str, Any]:
        sqlite_stat = db_path.stat()
        sqlite_info: dict[str, Any] = {
            "path": str(db_path),
            "size_bytes": sqlite_stat.st_size,
            "modified_at": datetime.fromtimestamp(sqlite_stat.st_mtime, UTC).isoformat(),
            "sha256": self._hash_file(db_path),
        }

        chroma_info: dict[str, Any] = {
            "path": str(chroma_path),
            "exists": chroma_path.exists(),
        }
        if chroma_path.exists():
            size_bytes = 0
            file_count = 0
            latest_mtime: float | None = None
            for entry in chroma_path.rglob("*"):
                if entry.is_symlink():
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                if entry.is_file():
                    size_bytes += stat.st_size
                    file_count += 1
                    latest_mtime = (
                        stat.st_mtime if latest_mtime is None else max(latest_mtime, stat.st_mtime)
                    )
            chroma_info["size_bytes"] = size_bytes
            chroma_info["files"] = file_count
            if latest_mtime is not None:
                chroma_info["modified_at"] = datetime.fromtimestamp(
                    latest_mtime,
                    UTC,
                ).isoformat()

        manifest: dict[str, Any] = {
            "created_at": datetime.now(UTC).isoformat(),
            "version": self._package_version(),
            "sqlite": sqlite_info,
            "chroma": chroma_info,
            "prompt_stats": self._serialise_catalogue_stats(),
        }
        return manifest

    def _write_chroma_directory(self, archive: zipfile.ZipFile, chroma_path: Path) -> None:
        if not chroma_path.exists():
            return
        for entry in chroma_path.rglob("*"):
            if entry.is_dir() or entry.is_symlink():
                continue
            arcname = Path("chroma") / entry.relative_to(chroma_path)
            archive.write(entry, arcname=str(arcname))

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _serialise_catalogue_stats(self) -> dict[str, Any] | None:
        try:
            stats = self.get_prompt_catalogue_stats()
        except PromptManagerError:
            return None
        payload = asdict(stats)
        last_modified = payload.get("last_modified_at")
        if isinstance(last_modified, datetime):
            payload["last_modified_at"] = last_modified.isoformat()
        return payload

    @staticmethod
    def _package_version() -> str | None:
        try:
            return importlib_metadata.version("prompt-manager")
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - dev installs
            return None
