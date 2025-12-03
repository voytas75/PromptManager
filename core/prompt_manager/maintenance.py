"""Maintenance mixin providing operational utilities for Prompt Manager.

Updates:
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
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
else:
    _PromptManager = Any

logger = logging.getLogger(__name__)

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
    """Operational helpers for repository, cache, and vector-store maintenance."""

    def get_redis_details(self: _PromptManager) -> dict[str, Any]:
        """Return connection and usage details for the configured Redis cache."""
        details: dict[str, Any] = {"enabled": self._redis_client is not None}
        client = self._redis_client
        if client is None:
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

    def get_chroma_details(self: _PromptManager) -> dict[str, Any]:
        """Return filesystem and collection metrics for the configured Chroma store."""
        details: dict[str, Any] = {"enabled": self._collection is not None}
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
        try:
            path_obj = Path(self._chroma_path)
            if path_obj.exists():
                size_bytes = sum(
                    entry.stat().st_size for entry in path_obj.rglob("*") if entry.is_file()
                )
                details.setdefault("stats", {})["disk_usage_bytes"] = size_bytes
        except (OSError, ValueError):  # pragma: no cover - filesystem specific
            pass
        return details

    def reset_prompt_repository(self: _PromptManager) -> None:
        """Clear all prompts, executions, and profiles from SQLite storage."""
        reset_func = getattr(self._repository, "reset_all_data", None)
        if not callable(reset_func):
            raise PromptManagerError("Repository reset is unavailable.")
        try:
            reset_func()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to reset prompt repository") from exc
        logger.info("Prompt repository reset completed.")
        self.refresh_user_profile()

    def reset_vector_store(self: _PromptManager) -> None:
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
        except Exception as exc:  # pragma: no cover - backend specific
            raise PromptStorageError("Unable to reset Chroma vector store") from exc
        logger.info("Chroma vector store reset completed.")

    def rebuild_embeddings(self: _PromptManager, *, reset_store: bool = False) -> tuple[int, int]:
        """Regenerate embeddings for all prompts and persist them to Chroma."""
        if reset_store:
            self.reset_vector_store()

        try:
            prompts = self._repository.list()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to load prompts for embedding rebuild") from exc

        if not prompts:
            logger.info("No prompts available for embedding rebuild.")
            return 0, 0

        successes = 0
        failures = 0
        for prompt in prompts:
            try:
                vector = self._embedding_provider.embed(prompt.document)
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
                self._repository.update(prompt)
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

            self._persist_embedding(prompt, vector, is_new=reset_store)
            successes += 1

        logger.info(
            "Embedding rebuild finished: %s succeeded, %s failed.",
            successes,
            failures,
        )
        return successes, failures

    def compact_vector_store(self: _PromptManager) -> None:
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

    def optimize_vector_store(self: _PromptManager) -> None:
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

    def verify_vector_store(self: _PromptManager) -> str:
        """Run integrity checks against the persistent Chroma store."""
        db_path = Path(self._chroma_path) / "chroma.sqlite3"
        if not db_path.exists():
            raise PromptStorageError(f"Chroma persistence database missing at {db_path}.")

        try:
            collection = self.collection
        except PromptManagerError:
            self._initialise_chroma_collection()
            try:
                collection = self.collection
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
        self._persist_chroma_client()
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
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

    def compact_repository(self: _PromptManager) -> None:
        """Vacuum the SQLite prompt repository to reclaim disk space."""
        db_path = self._resolve_repository_path()
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                connection.execute("VACUUM;")
        except sqlite3.Error as exc:
            raise PromptStorageError("Unable to compact SQLite repository") from exc
        logger.info("SQLite repository VACUUM completed at %s", db_path)

    def optimize_repository(self: _PromptManager) -> None:
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

    def verify_repository(self: _PromptManager) -> str:
        """Run integrity checks against the SQLite prompt repository."""
        db_path = self._resolve_repository_path()
        diagnostics: list[str] = []
        try:
            with sqlite3.connect(str(db_path), timeout=60.0) as connection:
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

    def create_data_snapshot(self: _PromptManager, destination: str | Path) -> Path:
        """Zip the SQLite repository, Chroma store, and a manifest for backups."""
        db_path = self._resolve_repository_path()
        chroma_path = Path(self._chroma_path).expanduser()
        self._persist_chroma_client()

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
                self._write_chroma_directory(archive, chroma_path)
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                )
        except OSError as exc:
            raise PromptManagerError(f"Unable to create snapshot archive: {exc}") from exc

        logger.info("Snapshot archive created at %s", archive_path)
        return archive_path

    def clear_usage_logs(
        self: _PromptManager, logs_path: str | Path | None = None
    ) -> None:
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

    def reset_application_data(
        self: _PromptManager, *, clear_logs: bool = True
    ) -> None:
        """Reset prompt data, embeddings, and optional usage logs."""
        self.reset_prompt_repository()
        self.reset_vector_store()
        if clear_logs:
            self.clear_usage_logs()

    def get_prompt_catalogue_stats(self: _PromptManager) -> PromptCatalogueStats:
        """Return aggregate prompt statistics for maintenance workflows."""
        try:
            return self._repository.get_prompt_catalogue_stats()
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
