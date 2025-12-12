"""noxfile.py - Nox sessions for Prompt Manager.

Updates:
  v0.3.0 - 2025-12-12 - Align sessions with Ruff/Pyright/Pytest quality gates in `.venv`.
  v0.2.3 - 2025-11-29 - Convert missing-tool warning to f-string for Ruff UP031.
  v0.2.2 - 2025-11-11 - Switch type checker session from mypy to pyright.
  v0.2.1 - 2025-11-07 - Document dev extra prerequisite for automation sessions.
  v0.2.0 - 2025-10-30 - Switch sessions to host interpreter and add tool detection
  v0.1.0 - 2025-10-30 - Initial scaffold of fmt/lint/tests/type-check sessions

Install the project with `pip install -e .[dev]` inside `.venv` before running these sessions.
This file defines automation sessions:
- format: format code with ruff
- lint: run ruff lint checks
- typecheck: run pyright in strict mode
- test: run pytest with coverage
- all: run the full quality gate suite

Adjust tool versions/args as needed. Sessions run directly in the host Python
environment (no isolated venv) but invoke tools from the project `.venv`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nox  # type: ignore[reportMissingImports]

    type Session = "nox.Session"
else:  # pragma: no cover - runtime fallback when nox is unavailable
    try:
        import nox as _nox

        nox = _nox
        Session = _nox.Session
    except ImportError:

        class _NoxStubSession:
            def run(self, *args, **kwargs) -> None:
                raise RuntimeError("nox is not installed; install the dev extras to run nox.")

            def error(self, message: str) -> None:
                raise RuntimeError(message)

        class _NoxStub:
            Session = _NoxStubSession

            def session(self, *args, **kwargs):  # type: ignore[override]
                def decorator(func):
                    return func

                return decorator

        nox = _NoxStub()
        Session = _NoxStubSession


CODE_LOCATIONS: tuple[str, ...] = (
    "main.py",
    "config",
    "core",
    "gui",
    "models",
    "tests",
)


def _venv_executable(command: str) -> Path:
    """Return the path to *command* inside the project virtual environment."""
    venv_dir = Path(".venv")
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{command}.exe"
    return venv_dir / "bin" / command


def _require_venv_tool(session: Session, command: str) -> str:
    """Return the `.venv` tool path, failing with guidance when missing."""
    candidate = _venv_executable(command)
    if candidate.exists():
        return str(candidate)
    session.error(
        "Project virtual environment tool is missing: "
        f"{candidate}. Create `.venv` and install dev tools with "
        "`python -m venv .venv && . .venv/bin/activate && pip install -e .[dev]`."
    )
    raise RuntimeError("unreachable")  # pragma: no cover


@nox.session(venv_backend="none")
def format(session: nox.Session) -> None:
    """Format code using ruff.

    Usage: `nox -s format`
    """
    ruff = _require_venv_tool(session, "ruff")
    session.run(ruff, "format", *CODE_LOCATIONS, external=True)


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    """Lint code using ruff.

    Usage: `nox -s lint`
    """
    ruff = _require_venv_tool(session, "ruff")
    session.run(ruff, "check", *CODE_LOCATIONS, external=True)


@nox.session(venv_backend="none")
def typecheck(session: nox.Session) -> None:
    """Run pyright in strict mode.

    Usage: `nox -s typecheck`
    """
    pyright = _require_venv_tool(session, "pyright")
    session.run(pyright, external=True)


@nox.session(venv_backend="none")
def test(session: nox.Session) -> None:
    """Run pytest with coverage.

    Usage: `nox -s test`
    """
    pytest = _require_venv_tool(session, "pytest")
    session.run(
        pytest,
        "-n",
        "auto",
        "--cov=core",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        *CODE_LOCATIONS,
        external=True,
    )


@nox.session(venv_backend="none")
def all(session: nox.Session) -> None:
    """Run the full Ruff/Pyright/Pytest quality gate suite.

    Usage: `nox -s all`
    """
    ruff = _require_venv_tool(session, "ruff")
    pyright = _require_venv_tool(session, "pyright")
    pytest = _require_venv_tool(session, "pytest")

    session.run(ruff, "check", "--fix", *CODE_LOCATIONS, external=True)
    session.run(ruff, "format", *CODE_LOCATIONS, external=True)
    session.run(ruff, "check", *CODE_LOCATIONS, external=True)
    session.run(ruff, "format", "--check", *CODE_LOCATIONS, external=True)
    session.run(pyright, external=True)
    session.run(
        pytest,
        "-n",
        "auto",
        "--cov=core",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        *CODE_LOCATIONS,
        external=True,
    )


@nox.session(venv_backend="none")
def fmt(session: nox.Session) -> None:
    """Alias for the format session."""
    format(session)


@nox.session(venv_backend="none")
def tests(session: nox.Session) -> None:
    """Alias for the test session."""
    test(session)


@nox.session(venv_backend="none")
def type_check(session: nox.Session) -> None:
    """Alias for the typecheck session."""
    typecheck(session)
