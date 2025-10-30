"""
noxfile.py - Nox sessions for Prompt Manager

Updates: v0.2.0 - 2025-10-30 - Switch sessions to host interpreter and add tool detection
Updates: v0.1.0 - 2025-10-30 - Initial scaffold of fmt/lint/tests/type-check sessions

This file defines automation sessions:
- fmt: format code with black
- lint: lint and check style with ruff (and black --check)
- tests: run pytest with coverage
- type_check: run mypy in strict mode over core modules

Adjust tool versions/args as needed. Sessions run directly in the host Python
environment (no isolated venv) so ensure required tools are installed locally.
"""

from __future__ import annotations

from typing import Iterable

import nox


def _ensure_tool(session: nox.Session, command: str, *version_args: str) -> None:
    """Verify that an external tool is available before executing commands."""

    args = (command, *(version_args or ("--version",)))
    try:
        session.run(*args, external=True)
    except Exception as exc:  # noqa: BLE001
        session.error(
            "Required tool '%s' is missing or not executable. Install it in the host "
            "environment before running this session. Original error: %s"
            % (command, exc)
        )

# Directories to operate on
CODE_LOCATIONS: tuple[str, ...] = (
    "main.py",
    "config",
    "core",
    "gui",
    "models",
    "tests",
)

@nox.session(venv_backend="none")
def fmt(session: nox.Session) -> None:
    """Format code using black.

    Usage: `nox -s fmt`
    """
    _ensure_tool(session, "black")
    session.run("black", *CODE_LOCATIONS, external=True)


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    """Lint code using ruff and check formatting with black --check.

    Usage: `nox -s lint`
    """
    _ensure_tool(session, "ruff")
    _ensure_tool(session, "black")
    # Ruff lint + fix unsafe disabled by default; change if you want autofixes
    session.run("ruff", "check", *CODE_LOCATIONS, external=True)
    session.run("black", "--check", *CODE_LOCATIONS, external=True)


@nox.session(venv_backend="none")
def tests(session: nox.Session) -> None:
    """Run pytest with coverage.

    Usage: `nox -s tests`
    """
    _ensure_tool(session, "pytest")
    # Add any env vars needed by tests here
    session.run(
        "pytest",
        "-q",
        "--cov=.",
        "--cov-report=term-missing",
        *CODE_LOCATIONS,
        external=True,
    )


@nox.session(venv_backend="none")
def type_check(session: nox.Session) -> None:
    """Run mypy in strict mode against core modules.

    Usage: `nox -s type_check`
    """
    _ensure_tool(session, "mypy")

    # Prefer a local mypy.ini/pyproject config if present; otherwise default to strict.
    mypy_args: list[str] = [
        "--strict",
        "--python-version",
        "3.11",
        "--show-error-codes",
        "--pretty",
    ]

    # Respect project config if found; mypy will auto-discover, so no extra flags needed.
    targets: Iterable[str] = (
        "main.py",
        "core",
        "config",
        "gui",
        "models",
        "tests",
    )

    session.run("mypy", *mypy_args, *targets, external=True)
