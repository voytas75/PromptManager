"""Analytics dashboard panel wiring for the Prompt Manager GUI.

Updates:
  v0.1.3 - 2025-12-05 - Prevent duplicate edit launches by using a single activation signal.
  v0.1.2 - 2025-12-05 - Make usage table prompts clickable to open the editor.
  v0.1.1 - 2025-11-29 - Wrap analytics strings to satisfy Ruff line-length rules.
  v0.1.0 - 2025-11-28 - Introduce dashboard tab with charts and CSV export.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QDateTimeAxis,
    QLineSeries,
    QValueAxis,
)
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import AnalyticsSnapshot, PromptManager, build_analytics_snapshot, snapshot_dataset_rows

from .processing_indicator import ProcessingIndicator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID


class AnalyticsDashboardPanel(QWidget):
    """Interactive analytics dashboard with charts and CSV export."""

    _DATASETS: Sequence[tuple[str, str]] = (
        ("usage", "Usage Frequency"),
        ("model_costs", "Model Cost Breakdown"),
        ("benchmark", "Benchmark Success"),
        ("intent", "Intent Success Trend"),
        ("embedding", "Embedding Health"),
    )

    def __init__(
        self,
        manager: PromptManager,
        parent: QWidget | None = None,
        *,
        usage_log_path: Path | None = None,
        prompt_edit_callback: Callable[[UUID], None] | None = None,
    ) -> None:
        """Initialise analytics widgets and load persisted preferences."""
        super().__init__(parent)
        self._manager = manager
        self._usage_log_path = usage_log_path
        self._snapshot: AnalyticsSnapshot | None = None
        self._chart_view = QChartView(self)
        self._dataset_combo = QComboBox(self)
        self._chart_type_combo = QComboBox(self)
        self._window_spin = QSpinBox(self)
        self._prompt_limit_spin = QSpinBox(self)
        self._table = QTableWidget(self)
        self._status_label = QLabel("", self)
        self._embedding_summary = QLabel("", self)
        self._settings = QSettings("PromptManager", "AnalyticsPanel")
        self._initial_window_days = self._settings.value("windowDays", 30, int)
        self._initial_prompt_limit = self._settings.value("promptLimit", 5, int)
        self._prompt_edit_callback = prompt_edit_callback
        self._usage_row_prompt_ids: dict[int, UUID] = {}
        self._build_ui()
        self._apply_initial_preferences()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            (
                "Explore execution, benchmark, and embedding metrics. "
                "Adjust the look-back window, dataset, and chart style to review performance."
            ),
            self,
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        dataset_label = QLabel("Dataset:", self)
        controls.addWidget(dataset_label)
        for key, label in self._DATASETS:
            self._dataset_combo.addItem(label, key)
        self._dataset_combo.currentIndexChanged.connect(self._update_visuals)  # type: ignore[arg-type]
        controls.addWidget(self._dataset_combo)

        controls.addWidget(QLabel("Chart:", self))
        self._chart_type_combo.addItem("Bar", "bar")
        self._chart_type_combo.addItem("Line", "line")
        self._chart_type_combo.currentIndexChanged.connect(self._update_chart)  # type: ignore[arg-type]
        controls.addWidget(self._chart_type_combo)

        controls.addWidget(QLabel("Window (days):", self))
        self._window_spin.setRange(0, 365)
        self._window_spin.setValue(30)
        self._window_spin.setMinimumWidth(90)
        self._window_spin.valueChanged.connect(self._handle_window_changed)  # type: ignore[arg-type]
        controls.addWidget(self._window_spin)

        controls.addWidget(QLabel("Top prompts:", self))
        self._prompt_limit_spin.setRange(3, 25)
        self._prompt_limit_spin.setValue(5)
        self._prompt_limit_spin.setMinimumWidth(90)
        self._prompt_limit_spin.valueChanged.connect(self._handle_prompt_limit_changed)  # type: ignore[arg-type]
        controls.addWidget(self._prompt_limit_spin)

        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self._handle_refresh_clicked)  # type: ignore[arg-type]
        controls.addWidget(refresh_button)

        export_button = QPushButton("Export CSV", self)
        export_button.clicked.connect(self._export_csv)  # type: ignore[arg-type]
        controls.addWidget(export_button)

        controls.addStretch(1)
        layout.addLayout(controls)

        self._chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self._chart_view, stretch=2)

        self._table.setColumnCount(0)
        self._table.setRowCount(0)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.cellActivated.connect(self._handle_table_cell_activated)  # type: ignore[arg-type]
        layout.addWidget(self._table, stretch=1)

        self._embedding_summary.setWordWrap(True)
        layout.addWidget(self._embedding_summary)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._status_label)
        status_layout.addStretch(1)
        layout.addLayout(status_layout)

    def refresh(self, *, show_indicator: bool = False) -> None:
        """Recompute analytics snapshot and update the chart/table views."""
        window_days = self._window_spin.value()
        prompt_limit = self._prompt_limit_spin.value()
        message = "Refreshing analytics…"

        def _build_snapshot() -> AnalyticsSnapshot:
            return build_analytics_snapshot(
                self._manager,
                window_days=window_days,
                prompt_limit=prompt_limit,
                usage_log_path=self._usage_log_path,
            )

        try:
            if show_indicator:
                snapshot = ProcessingIndicator(self, message, title="Updating Analytics").run(
                    _build_snapshot
                )
            else:
                snapshot = _build_snapshot()
        except Exception as exc:  # pragma: no cover - GUI feedback only
            QMessageBox.critical(self, "Analytics", f"Unable to build analytics snapshot: {exc}")
            return
        self._snapshot = snapshot
        self._update_visuals()
        now_local = datetime.now(UTC).astimezone()
        self._status_label.setText(f"Refreshed at {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self._persist_preferences(window_days, prompt_limit)

    def _handle_window_changed(self, _: int) -> None:
        self.refresh(show_indicator=True)

    def _handle_prompt_limit_changed(self, _: int) -> None:
        self.refresh(show_indicator=True)

    def _handle_refresh_clicked(self) -> None:
        self.refresh(show_indicator=True)

    def _update_visuals(self) -> None:
        self._update_chart()
        self._populate_table()
        self._update_embedding_summary()

    def _update_chart(self) -> None:
        chart = QChart()
        chart.legend().setVisible(False)
        dataset_key = self._dataset_combo.currentData()
        chart_type = self._chart_type_combo.currentData()
        snapshot = self._snapshot
        if snapshot is None:
            self._chart_view.setChart(chart)
            return

        if dataset_key == "embedding":
            chart.setTitle("Embedding health visualised in the summary below")
            self._chart_view.setChart(chart)
            return

        labels: list[str] = []
        values: list[float] = []
        if dataset_key == "usage":
            for entry in snapshot.usage_frequency:
                labels.append(entry.name)
                values.append(float(entry.usage_count))
            chart.setTitle("Prompt usage frequency")
        elif dataset_key == "model_costs":
            for entry in snapshot.model_costs:
                labels.append(entry.model)
                values.append(float(entry.total_tokens))
            chart.setTitle("Total tokens by model")
        elif dataset_key == "benchmark":
            for entry in snapshot.benchmark_stats:
                labels.append(entry.model)
                values.append(entry.success_rate * 100)
            chart.setTitle("Benchmark success rate (%)")
        elif dataset_key == "intent":
            line_series = QLineSeries()
            for point in snapshot.intent_success:
                labels.append(point.bucket.date().isoformat())
                line_series.append(point.bucket.timestamp() * 1000, point.success_rate * 100)
            if not labels:
                chart.setTitle("No intent success data recorded")
                self._chart_view.setChart(chart)
                return
            chart.addSeries(line_series)
            axis_x = QDateTimeAxis()
            axis_x.setFormat("MMM d")
            axis_x.setTitleText("Date")
            axis_y = QValueAxis()
            axis_y.setRange(0, 100)
            axis_y.setTitleText("Success %")
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            line_series.attachAxis(axis_x)
            line_series.attachAxis(axis_y)
            chart.setTitle("Intent workspace success rate")
            self._chart_view.setChart(chart)
            return
        else:
            self._chart_view.setChart(chart)
            return

        if not labels:
            chart.setTitle("No data available for this dataset")
            self._chart_view.setChart(chart)
            return

        if chart_type == "line":
            series = QLineSeries()
            for index, value in enumerate(values):
                series.append(float(index), value)
            chart.addSeries(series)
            axis_x = QValueAxis()
            axis_x.setRange(0, max(len(values) - 1, 0))
            axis_x.setLabelFormat("%d")
            axis_x.setTitleText("Index")
            axis_y = QValueAxis()
            axis_y.setRange(0, max(values) * 1.1 if values else 1)
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
        else:
            bar_series = QBarSeries()
            bar_set = QBarSet("value")
            for value in values:
                bar_set << value
            bar_series.append(bar_set)
            chart.addSeries(bar_series)
            axis_x = QBarCategoryAxis()
            axis_x.append(labels)
            axis_y = QValueAxis()
            axis_y.setRange(0, max(values) * 1.1 if values else 1)
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            bar_series.attachAxis(axis_x)
            bar_series.attachAxis(axis_y)

        self._chart_view.setChart(chart)

    def _populate_table(self) -> None:
        dataset_key = self._dataset_combo.currentData()
        snapshot = self._snapshot
        if snapshot is None:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return

        headers: list[str]
        rows: list[list[str]]
        self._usage_row_prompt_ids.clear()
        if dataset_key == "usage":
            headers = ["Prompt", "Usage Count", "Success Rate", "Last Used"]
            rows = []
            for index, entry in enumerate(snapshot.usage_frequency):
                last_used = (
                    entry.last_executed_at.isoformat(timespec="seconds")
                    if entry.last_executed_at
                    else "n/a"
                )
                rows.append(
                    [
                        entry.name,
                        str(entry.usage_count),
                        self._format_pct(entry.success_rate),
                        last_used,
                    ]
                )
                self._usage_row_prompt_ids[index] = entry.prompt_id
        elif dataset_key == "model_costs":
            headers = ["Model", "Runs", "Prompt Tokens", "Completion Tokens", "Total Tokens"]
            rows = [
                [
                    entry.model,
                    str(entry.run_count),
                    str(entry.prompt_tokens),
                    str(entry.completion_tokens),
                    str(entry.total_tokens),
                ]
                for entry in snapshot.model_costs
            ]
        elif dataset_key == "benchmark":
            headers = ["Model", "Runs", "Success Rate", "Avg Duration (ms)", "Tokens"]
            rows = []
            for entry in snapshot.benchmark_stats:
                duration = (
                    f"{entry.average_duration_ms:.0f}"
                    if entry.average_duration_ms is not None
                    else "n/a"
                )
                rows.append(
                    [
                        entry.model,
                        str(entry.run_count),
                        self._format_pct(entry.success_rate),
                        duration,
                        str(entry.total_tokens),
                    ]
                )
        elif dataset_key == "intent":
            headers = ["Date", "Success Rate", "Success", "Total"]
            rows = [
                [
                    point.bucket.date().isoformat(),
                    self._format_pct(point.success_rate),
                    str(point.success),
                    str(point.total),
                ]
                for point in snapshot.intent_success
            ]
        else:
            headers = ["Metric", "Value"]
            report = snapshot.embedding
            if report is None:
                rows = [["Status", "Embedding diagnostics unavailable"]]
            else:
                backend_status = "ok" if report.backend_ok else "error"
                chroma_status = "ok" if report.chroma_ok else "error"
                dimension = report.backend_dimension or report.inferred_dimension or "n/a"
                rows = [
                    ["Backend", f"{backend_status} - {report.backend_message}"],
                    ["Dimension", str(dimension)],
                    ["Chroma", f"{chroma_status} - {report.chroma_message}"],
                    ["Stored count", str(report.prompts_with_embeddings)],
                    ["Repository prompts", str(report.repository_total)],
                    ["Missing", str(len(report.missing_prompts))],
                    ["Dimension mismatches", str(len(report.mismatched_prompts))],
                ]
            self._usage_row_prompt_ids.clear()

        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self._table.setItem(row_index, column_index, QTableWidgetItem(value))
        self._table.resizeColumnsToContents()

    def _handle_table_cell_activated(self, row: int, _: int) -> None:
        self._activate_usage_row(row)

    def _activate_usage_row(self, row: int) -> None:
        if self._dataset_combo.currentData() != "usage":
            return
        if self._prompt_edit_callback is None:
            return
        prompt_id = self._usage_row_prompt_ids.get(row)
        if prompt_id is None:
            return
        self._prompt_edit_callback(prompt_id)

    def _update_embedding_summary(self) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.embedding is None:
            self._embedding_summary.setText("Embedding diagnostics unavailable.")
            return
        report = snapshot.embedding
        consistent = (
            "consistent"
            if report.consistent_counts
            else "mismatch"
            if report.consistent_counts is False
            else "unknown"
        )
        backend_descriptor = "ok" if report.backend_ok else "error"
        chroma_descriptor = "ok" if report.chroma_ok else "error"
        vector_summary = (
            f"Vectors stored {report.prompts_with_embeddings}/"
            f"{report.repository_total} ({consistent})."
        )
        summary = (
            f"Embedding backend: {backend_descriptor} ({report.backend_message}). "
            f"Chroma: {chroma_descriptor} ({report.chroma_message}). "
            f"{vector_summary}"
        )
        self._embedding_summary.setText(summary)

    def _export_csv(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            QMessageBox.information(self, "Analytics", "No analytics data available to export.")
            return
        dataset_key = self._dataset_combo.currentData()
        try:
            rows = snapshot_dataset_rows(snapshot, dataset_key)
        except ValueError as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Analytics", str(exc))
            return
        if not rows:
            QMessageBox.information(self, "Analytics", "Selected dataset has no rows to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export analytics dataset",
            "analytics.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            self._write_csv(Path(path), rows)
        except OSError as exc:  # pragma: no cover - filesystem errors rare in tests
            QMessageBox.critical(self, "Analytics", f"Unable to export dataset: {exc}")
            return
        QMessageBox.information(self, "Analytics", f"Dataset exported to {path}")

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
        headers: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key in seen:
                    continue
                seen.add(key)
                headers.append(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({header: row.get(header, "") for header in headers})

    @staticmethod
    def _format_pct(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%"

    def _apply_initial_preferences(self) -> None:
        self._window_spin.blockSignals(True)
        self._prompt_limit_spin.blockSignals(True)
        window_pref = max(0, min(365, int(self._initial_window_days)))
        prompt_pref = max(3, min(25, int(self._initial_prompt_limit)))
        self._window_spin.setValue(window_pref)
        self._prompt_limit_spin.setValue(prompt_pref)
        self._window_spin.blockSignals(False)
        self._prompt_limit_spin.blockSignals(False)

    def _persist_preferences(self, window_days: int, prompt_limit: int) -> None:
        self._settings.setValue("windowDays", window_days)
        self._settings.setValue("promptLimit", prompt_limit)


__all__ = ["AnalyticsDashboardPanel"]
