"""Offline regressions for the workspace web-search preference."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox

from config import load_settings
from gui.main_window import MainWindow
from gui.runtime_settings_service import RuntimeSettingsService
from gui.settings_dialog import SettingsDialog

if TYPE_CHECKING:
    from pathlib import Path


def _no_provider_setup(*args: Any, **kwargs: Any) -> None:
    """Replace only the provider setup seam, not persistence or hydration."""


@pytest.mark.parametrize("enabled", [False, True])
def test_workspace_preference_survives_settings_save_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """A toggle survives a real settings-dialog payload save and fresh hydration."""
    for key in list(os.environ):
        if key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_")):
            monkeypatch.delenv(key)
    config_path = tmp_path / "active.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))
    monkeypatch.setenv("PROMPT_MANAGER_ENV_FILE", "")
    app = QApplication.instance() or QApplication([])
    manager = SimpleNamespace(
        executor=None,
        set_name_generator=_no_provider_setup,
        redis_unavailable_reason=None,
        _redis_client=None,
    )
    settings = load_settings()
    service = RuntimeSettingsService(cast("Any", manager), settings)
    runtime = service.build_initial_runtime_settings()
    checkbox = QCheckBox()
    window = SimpleNamespace(
        _runtime_settings=runtime, _settings=settings, _web_search_checkbox=checkbox
    )
    MainWindow._bind_web_search_preference(cast("Any", window))  # pyright: ignore[reportPrivateUsage]
    assert checkbox.isChecked() is True
    # Exercise both edges even for the default-ON parametrization.
    checkbox.setChecked(not enabled)
    checkbox.setChecked(enabled)
    assert json.loads(config_path.read_text())["use_web_search"] is enabled
    assert runtime["use_web_search"] is enabled
    dialog = SettingsDialog(web_search_provider="random")
    try:
        service.apply_updates(runtime, dialog.result_settings())
    finally:
        dialog.close()
    assert json.loads(config_path.read_text())["use_web_search"] is enabled
    reopened = SettingsDialog(web_search_provider=cast("Any", runtime["web_search_provider"]))
    try:
        assert reopened.result_settings()["web_search_provider"] == "random"
    finally:
        reopened.close()
    fresh_settings = load_settings()
    fresh_runtime = RuntimeSettingsService(
        cast("Any", manager), fresh_settings
    ).build_initial_runtime_settings()
    fresh_checkbox = QCheckBox()
    fresh_window = SimpleNamespace(
        _runtime_settings=fresh_runtime,
        _settings=fresh_settings,
        _web_search_checkbox=fresh_checkbox,
    )
    before = config_path.read_bytes()
    MainWindow._bind_web_search_preference(cast("Any", fresh_window))  # pyright: ignore[reportPrivateUsage]
    assert fresh_checkbox.isChecked() is enabled
    assert config_path.read_bytes() == before
    assert app is not None


@pytest.mark.parametrize("enabled", [False, True])
def test_preference_binding_without_settings_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """Runtime-only windows preserve both toggle values without a settings model."""
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(tmp_path / "config.json"))
    app = QApplication.instance() or QApplication([])
    checkbox = QCheckBox()
    runtime: dict[str, object | None] = {"use_web_search": enabled}
    window = SimpleNamespace(
        _runtime_settings=runtime, _settings=None, _web_search_checkbox=checkbox
    )
    MainWindow._bind_web_search_preference(cast("Any", window))  # pyright: ignore[reportPrivateUsage]
    assert checkbox.isChecked() is enabled
    checkbox.setChecked(not enabled)
    assert runtime["use_web_search"] is not enabled
    assert app is not None
