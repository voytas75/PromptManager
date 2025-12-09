"""Backend maintenance helpers for Redis, ChromaDB, and SQLite tabs.

Updates:
  v0.1.1 - 2025-12-09 - Show Redis cache availability banner with clear contrast styling.
  v0.1.0 - 2025-12-04 - Extract backend tab builders and maintenance routines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import PromptManager, PromptManagerError, RepositoryError

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from PySide6.QtWidgets import QWidget as _QWidget


class BackendMaintenanceMixin:
    """Provide backend inspection and maintenance utilities."""

    _manager: PromptManager
    _redis_status_label: QLabel
    _redis_connection_label: QLabel
    _redis_banner_label: QLabel
    _redis_stats_view: QPlainTextEdit
    _redis_refresh_button: QPushButton
    _chroma_status_label: QLabel
    _chroma_path_label: QLabel
    _chroma_stats_view: QPlainTextEdit
    _chroma_refresh_button: QPushButton
    _chroma_compact_button: QPushButton
    _chroma_optimize_button: QPushButton
    _chroma_verify_button: QPushButton
    _storage_status_label: QLabel
    _storage_path_label: QLabel
    _storage_stats_view: QPlainTextEdit
    _storage_refresh_button: QPushButton
    _sqlite_compact_button: QPushButton
    _sqlite_optimize_button: QPushButton
    _sqlite_verify_button: QPushButton

    def _build_redis_tab(self, parent: QWidget) -> QWidget:
        redis_tab = QWidget(parent)
        redis_layout = QVBoxLayout(redis_tab)

        redis_description = QLabel("Inspect the Redis cache used for prompt caching.", redis_tab)
        redis_description.setWordWrap(True)
        redis_layout.addWidget(redis_description)

        self._redis_banner_label = QLabel("", redis_tab)
        self._redis_banner_label.setWordWrap(True)
        self._redis_banner_label.setVisible(False)
        redis_layout.addWidget(self._redis_banner_label)

        status_container = QWidget(redis_tab)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        self._redis_status_label = QLabel("Checking…", status_container)
        status_layout.addWidget(self._redis_status_label)

        self._redis_connection_label = QLabel("", status_container)
        self._redis_connection_label.setWordWrap(True)
        status_layout.addWidget(self._redis_connection_label, stretch=1)

        self._redis_refresh_button = QPushButton("Refresh", status_container)
        self._redis_refresh_button.clicked.connect(self._refresh_redis_info)  # type: ignore[arg-type]
        status_layout.addWidget(self._redis_refresh_button)

        redis_layout.addWidget(status_container)

        self._redis_stats_view = QPlainTextEdit(redis_tab)
        self._redis_stats_view.setReadOnly(True)
        redis_layout.addWidget(self._redis_stats_view, stretch=1)

        return redis_tab

    def _build_chroma_tab(self, parent: QWidget) -> QWidget:
        chroma_tab = QWidget(parent)
        chroma_layout = QVBoxLayout(chroma_tab)

        chroma_description = QLabel(
            "Review the ChromaDB vector store used for semantic search.", chroma_tab
        )
        chroma_description.setWordWrap(True)
        chroma_layout.addWidget(chroma_description)

        chroma_status_container = QWidget(chroma_tab)
        chroma_status_layout = QHBoxLayout(chroma_status_container)
        chroma_status_layout.setContentsMargins(0, 0, 0, 0)
        chroma_status_layout.setSpacing(12)

        self._chroma_status_label = QLabel("Checking…", chroma_status_container)
        chroma_status_layout.addWidget(self._chroma_status_label)

        self._chroma_path_label = QLabel("", chroma_status_container)
        self._chroma_path_label.setWordWrap(True)
        chroma_status_layout.addWidget(self._chroma_path_label, stretch=1)

        self._chroma_refresh_button = QPushButton("Refresh", chroma_status_container)
        self._chroma_refresh_button.clicked.connect(self._refresh_chroma_info)  # type: ignore[arg-type]
        chroma_status_layout.addWidget(self._chroma_refresh_button)

        chroma_layout.addWidget(chroma_status_container)

        self._chroma_stats_view = QPlainTextEdit(chroma_tab)
        self._chroma_stats_view.setReadOnly(True)
        chroma_layout.addWidget(self._chroma_stats_view, stretch=1)

        chroma_actions_container = QWidget(chroma_tab)
        chroma_actions_layout = QHBoxLayout(chroma_actions_container)
        chroma_actions_layout.setContentsMargins(0, 0, 0, 0)
        chroma_actions_layout.setSpacing(12)

        self._chroma_compact_button = QPushButton(
            "Compact Persistent Store",
            chroma_actions_container,
        )
        self._chroma_compact_button.setToolTip(
            "Reclaim disk space by vacuuming the Chroma SQLite store."
        )
        self._chroma_compact_button.clicked.connect(self._on_chroma_compact_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_compact_button)

        self._chroma_optimize_button = QPushButton(
            "Optimize Persistent Store",
            chroma_actions_container,
        )
        self._chroma_optimize_button.setToolTip(
            "Refresh query statistics to improve Chroma performance."
        )
        self._chroma_optimize_button.clicked.connect(self._on_chroma_optimize_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_optimize_button)

        self._chroma_verify_button = QPushButton(
            "Verify Index Integrity",
            chroma_actions_container,
        )
        self._chroma_verify_button.setToolTip(
            "Run integrity checks against the Chroma index files."
        )
        self._chroma_verify_button.clicked.connect(self._on_chroma_verify_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_verify_button)

        chroma_actions_layout.addStretch(1)
        chroma_layout.addWidget(chroma_actions_container)

        return chroma_tab

    def _build_storage_tab(self, parent: QWidget) -> QWidget:
        storage_tab = QWidget(parent)
        storage_layout = QVBoxLayout(storage_tab)

        storage_description = QLabel(
            "Inspect the SQLite repository backing prompt storage.", storage_tab
        )
        storage_description.setWordWrap(True)
        storage_layout.addWidget(storage_description)

        storage_status_container = QWidget(storage_tab)
        storage_status_layout = QHBoxLayout(storage_status_container)
        storage_status_layout.setContentsMargins(0, 0, 0, 0)
        storage_status_layout.setSpacing(12)

        self._storage_status_label = QLabel("Checking…", storage_status_container)
        storage_status_layout.addWidget(self._storage_status_label)

        self._storage_path_label = QLabel("", storage_status_container)
        self._storage_path_label.setWordWrap(True)
        storage_status_layout.addWidget(self._storage_path_label, stretch=1)

        self._storage_refresh_button = QPushButton("Refresh", storage_status_container)
        self._storage_refresh_button.clicked.connect(self._refresh_storage_info)  # type: ignore[arg-type]
        storage_status_layout.addWidget(self._storage_refresh_button)

        storage_layout.addWidget(storage_status_container)

        self._storage_stats_view = QPlainTextEdit(storage_tab)
        self._storage_stats_view.setReadOnly(True)
        storage_layout.addWidget(self._storage_stats_view, stretch=1)

        storage_actions_container = QWidget(storage_tab)
        storage_actions_layout = QHBoxLayout(storage_actions_container)
        storage_actions_layout.setContentsMargins(0, 0, 0, 0)
        storage_actions_layout.setSpacing(12)

        self._sqlite_compact_button = QPushButton(
            "Compact Database",
            storage_actions_container,
        )
        self._sqlite_compact_button.setToolTip(
            "Run VACUUM on the prompt database to reclaim space."
        )
        self._sqlite_compact_button.clicked.connect(self._on_sqlite_compact_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_compact_button)

        self._sqlite_optimize_button = QPushButton("Optimize Database", storage_actions_container)
        self._sqlite_optimize_button.setToolTip("Refresh SQLite statistics for prompt lookups.")
        self._sqlite_optimize_button.clicked.connect(self._on_sqlite_optimize_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_optimize_button)

        self._sqlite_verify_button = QPushButton(
            "Verify Index Integrity",
            storage_actions_container,
        )
        self._sqlite_verify_button.setToolTip(
            "Run integrity checks against the prompt database indexes."
        )
        self._sqlite_verify_button.clicked.connect(self._on_sqlite_verify_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_verify_button)

        storage_actions_layout.addStretch(1)
        storage_layout.addWidget(storage_actions_container)

        return storage_tab

    def _set_chroma_actions_busy(self, busy: bool) -> None:
        if not busy:
            return
        for button in (
            self._chroma_refresh_button,
            self._chroma_compact_button,
            self._chroma_optimize_button,
            self._chroma_verify_button,
        ):
            button.setEnabled(False)

    def _parent_widget(self) -> "_QWidget":
        return cast("_QWidget", self)

    def _on_chroma_compact_clicked(self) -> None:
        self._set_chroma_actions_busy(True)
        parent = self._parent_widget()
        try:
            self._manager.compact_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(parent, "Compaction failed", str(exc))
            return
        else:
            QMessageBox.information(
                parent,
                "Chroma store compacted",
                "The persistent Chroma store has been vacuumed and reclaimed space.",
            )
        finally:
            self._refresh_chroma_info()

    def _on_chroma_optimize_clicked(self) -> None:
        self._set_chroma_actions_busy(True)
        parent = self._parent_widget()
        try:
            self._manager.optimize_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(parent, "Optimization failed", str(exc))
            return
        else:
            QMessageBox.information(
                parent,
                "Chroma store optimized",
                "Chroma query statistics have been refreshed for better performance.",
            )
        finally:
            self._refresh_chroma_info()

    def _on_chroma_verify_clicked(self) -> None:
        self._set_chroma_actions_busy(True)
        parent = self._parent_widget()
        try:
            summary = self._manager.verify_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(parent, "Verification failed", str(exc))
            return
        else:
            message = summary or "Chroma store integrity verified successfully."
            QMessageBox.information(parent, "Chroma store verified", message)
        finally:
            self._refresh_chroma_info()

    def _set_storage_actions_busy(self, busy: bool) -> None:
        if not busy:
            return
        for button in (
            self._storage_refresh_button,
            self._sqlite_compact_button,
            self._sqlite_optimize_button,
            self._sqlite_verify_button,
        ):
            button.setEnabled(False)

    def _on_sqlite_compact_clicked(self) -> None:
        self._set_storage_actions_busy(True)
        parent = self._parent_widget()
        try:
            self._manager.compact_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(parent, "Compaction failed", str(exc))
            return
        else:
            QMessageBox.information(
                parent,
                "Prompt database compacted",
                "The SQLite repository has been vacuumed and reclaimed space.",
            )
        finally:
            self._refresh_storage_info()

    def _on_sqlite_optimize_clicked(self) -> None:
        self._set_storage_actions_busy(True)
        parent = self._parent_widget()
        try:
            self._manager.optimize_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(parent, "Optimization failed", str(exc))
            return
        else:
            QMessageBox.information(
                parent,
                "Prompt database optimized",
                "SQLite statistics have been refreshed for prompt lookups.",
            )
        finally:
            self._refresh_storage_info()

    def _on_sqlite_verify_clicked(self) -> None:
        self._set_storage_actions_busy(True)
        parent = self._parent_widget()
        try:
            summary = self._manager.verify_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(parent, "Verification failed", str(exc))
            return
        else:
            message = summary or "SQLite repository integrity verified successfully."
            QMessageBox.information(parent, "Prompt database verified", message)
        finally:
            self._refresh_storage_info()

    def _refresh_redis_info(self) -> None:
        details = self._manager.get_redis_details()
        enabled = details.get("enabled", False)
        reason = details.get("reason")
        error_text = details.get("error")

        def _set_banner(text: str, *, intent: str) -> None:
            self._redis_banner_label.setText(text)
            self._redis_banner_label.setVisible(True)
            if intent == "error":
                bg = "#fdecea"
                border = "#f5c2c7"
                color = "#7a1c1c"
            elif intent == "warn":
                bg = "#fff4e5"
                border = "#f0ad4e"
                color = "#8a4b0f"
            else:
                bg = "#e8f5e9"
                border = "#2e7d32"
                color = "#1b5e20"
            self._redis_banner_label.setStyleSheet(
                f"background-color: {bg}; border: 1px solid {border}; "
                f"border-radius: 6px; padding: 8px; color: {color};"
            )

        if not enabled:
            self._redis_status_label.setText("Redis caching is disabled.")
            self._redis_connection_label.setText("")
            self._redis_stats_view.setPlainText("")
            self._redis_refresh_button.setEnabled(False)
            if reason:
                _set_banner(reason, intent="warn")
            else:
                self._redis_banner_label.setVisible(False)
            return

        self._redis_refresh_button.setEnabled(True)
        self._redis_banner_label.setVisible(False)

        status = details.get("status", "unknown").capitalize()
        if details.get("error"):
            status = f"Error: {details['error']}"
        self._redis_status_label.setText(f"Status: {status}")

        connection = details.get("connection", {})
        connection_parts: list[str] = []
        if connection.get("host"):
            host = connection["host"]
            port = connection.get("port")
            if port is not None:
                connection_parts.append(f"{host}:{port}")
            else:
                connection_parts.append(str(host))
        if connection.get("database") is not None:
            connection_parts.append(f"DB {connection['database']}")
        if connection.get("ssl"):
            connection_parts.append("SSL")
        if not connection_parts:
            self._redis_connection_label.setText("")
        else:
            self._redis_connection_label.setText("Connection: " + ", ".join(connection_parts))
        if details.get("status") == "error" and error_text:
            _set_banner(f"Redis connection error: {error_text}", intent="error")
        elif status.lower() == "online":
            _set_banner("Redis caching enabled and reachable.", intent="success")

        stats = details.get("stats", {})
        lines: list[str] = []
        for key, label in (
            ("keys", "Keys"),
            ("used_memory_human", "Used memory"),
            ("used_memory_peak_human", "Peak memory"),
            ("maxmemory_human", "Configured max memory"),
            ("hits", "Keyspace hits"),
            ("misses", "Keyspace misses"),
            ("hit_rate", "Hit rate (%)"),
        ):
            if stats.get(key) is not None:
                lines.append(f"{label}: {stats[key]}")
        if not lines and details.get("error"):
            lines.append(details["error"])
        elif not lines and stats.get("info_error"):
            lines.append(f"Unable to fetch stats: {stats['info_error']}")
        redis_text = "\n".join(lines) if lines else "No Redis statistics available."
        self._redis_stats_view.setPlainText(redis_text)

    def _refresh_chroma_info(self) -> None:
        details = self._manager.get_chroma_details()
        enabled = details.get("enabled", False)
        path = details.get("path") or ""
        collection = details.get("collection") or ""
        if not enabled:
            self._chroma_status_label.setText("ChromaDB is not initialised.")
            self._chroma_path_label.setText(f"Path: {path}" if path else "")
            self._chroma_stats_view.setPlainText("")
            self._chroma_refresh_button.setEnabled(False)
            self._chroma_compact_button.setEnabled(False)
            self._chroma_optimize_button.setEnabled(False)
            self._chroma_verify_button.setEnabled(False)
            return

        self._chroma_refresh_button.setEnabled(True)

        status = details.get("status", "unknown").capitalize()
        if details.get("error"):
            status = f"Error: {details['error']}"
        self._chroma_status_label.setText(f"Status: {status}")

        has_error = bool(details.get("error"))
        self._chroma_compact_button.setEnabled(not has_error)
        self._chroma_optimize_button.setEnabled(not has_error)
        self._chroma_verify_button.setEnabled(not has_error)

        path_parts: list[str] = []
        if path:
            path_parts.append(f"Path: {path}")
        if collection:
            path_parts.append(f"Collection: {collection}")
        self._chroma_path_label.setText(" | ".join(path_parts))

        stats = details.get("stats", {})
        lines: list[str] = []
        for key, label in (
            ("documents", "Documents"),
            ("disk_usage_bytes", "Disk usage (bytes)"),
        ):
            value = stats.get(key)
            if value is not None:
                lines.append(f"{label}: {value}")
        chroma_text = "\n".join(lines) if lines else "No ChromaDB statistics available."
        self._chroma_stats_view.setPlainText(chroma_text)

    def _refresh_storage_info(self) -> None:
        repository = self._manager.repository
        db_path_obj = getattr(repository, "_db_path", None)
        if isinstance(db_path_obj, Path):
            db_path = str(db_path_obj)
        else:
            db_path = str(db_path_obj) if db_path_obj is not None else ""

        self._storage_path_label.setText(f"Path: {db_path}" if db_path else "Path: unknown")
        self._storage_refresh_button.setEnabled(True)

        stats_lines: list[str] = []
        healthy = True

        size_bytes = None
        if db_path:
            try:
                path_obj = Path(db_path)
                if path_obj.exists():
                    size_bytes = path_obj.stat().st_size
                else:
                    healthy = False
                    stats_lines.append("Database file not found.")
            except OSError as exc:
                healthy = False
                stats_lines.append(f"File size: error ({exc})")
        else:
            healthy = False

        if size_bytes is not None:
            stats_lines.append(f"File size: {size_bytes} bytes")

        try:
            prompt_count = len(repository.list())
            stats_lines.append(f"Prompts: {prompt_count}")
        except RepositoryError as exc:
            healthy = False
            stats_lines.append(f"Prompts: error ({exc})")

        try:
            execution_count = len(repository.list_executions())
            stats_lines.append(f"Executions: {execution_count}")
        except RepositoryError as exc:
            healthy = False
            stats_lines.append(f"Executions: error ({exc})")

        storage_text = "\n".join(stats_lines) if stats_lines else "No SQLite statistics available."
        self._storage_stats_view.setPlainText(storage_text)

        if healthy:
            self._storage_status_label.setText("Status: ready")
        else:
            self._storage_status_label.setText("Status: unavailable")

        self._sqlite_compact_button.setEnabled(healthy)
        self._sqlite_optimize_button.setEnabled(healthy)
        self._sqlite_verify_button.setEnabled(healthy)
