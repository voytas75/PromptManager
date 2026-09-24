"""Exercise the real GUI entrypoint without provider credentials or user data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_offline_gui_capture_reuse_and_run_guard(tmp_path: Path) -> None:
    """An unconfigured model must not block local GUI work or permit a model call."""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_"))
            or key in {"LITELLM_API_KEY", "OPENAI_API_KEY"}
        )
    }
    env.update(
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / "xdg_config"),
        XDG_DATA_HOME=str(tmp_path / "xdg_data"),
        QT_QPA_PLATFORM="offscreen",
        ANONYMIZED_TELEMETRY="False",
        PYTHONDONTWRITEBYTECODE="1",
    )
    script = """
import sys
import traceback
from unittest.mock import patch
from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit
import main
from gui.main_window import MainWindow
from gui.dialogs.quick_capture import QuickCaptureDialog

errors = []

def verify():
    try:
        window = next(w for w in QApplication.topLevelWidgets() if isinstance(w, MainWindow))
        manager = window._manager
        assert window.isVisible() and manager.llm_available is False and manager.executor is None
        flow = window._prompt_editor_flow
        assert flow is not None

        def accept_capture(dialog):
            body = dialog.findChild(QPlainTextEdit, "quickCaptureBodyInput")
            assert body is not None
            body.setPlainText("Write a compact review of this draft.")
            dialog._on_accept()
            return QDialog.DialogCode.Accepted

        with patch.object(QuickCaptureDialog, "exec", accept_capture):
            with patch.object(flow, "edit_prompt", lambda _: None):
                flow.quick_capture_prompt()
        prompts = manager.repository.list()
        assert len(prompts) == 1
        prompt = prompts[0]
        assert prompt.context == "Write a compact review of this draft."
        assert (prompt.ext2 or {}).get("capture_state") == "draft"
        assert window._current_prompt() is not None
        window._prompt_actions_bridge.copy_prompt()
        assert QGuiApplication.clipboard().text() == prompt.context
        window._prompt_actions_bridge.open_prompt_in_workspace()
        assert window._query_input.toPlainText() == prompt.context
        controller = window._execution_controller
        assert controller is not None
        messages = []
        with patch.object(manager, "execute_prompt", side_effect=AssertionError("model called")):
            with patch.object(
                controller, "_error", lambda title, msg: messages.append((title, msg))
            ):
                with patch.object(controller, "_status", lambda *_: None):
                    controller.execute_prompt_with_text(
                        prompt,
                        "test input",
                        status_prefix="Ran",
                        empty_text_message="Empty",
                        keep_text_after=True,
                    )
        assert len(messages) == 1 and messages[0][0] == "Prompt execution unavailable"
        window.close()
    except Exception:
        errors.append(traceback.format_exc())
    finally:
        QApplication.instance().quit()

QTimer.singleShot(0, verify)
exit_code = main.main()
if errors:
    print(errors[0], file=sys.stderr)
    raise SystemExit(1)
assert exit_code == 0
print("OFFLINE_GUI_OPERATOR_PATH_OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-500:] + completed.stderr[-1500:]
    assert "OFFLINE_GUI_OPERATOR_PATH_OK" in completed.stdout
    assert (tmp_path / "catalog.db").is_file()
