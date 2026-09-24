"""Lightweight console-script entrypoint for provider-free doctor startup."""

from __future__ import annotations


def main() -> int:
    """Dispatch doctor before importing the provider-backed application runtime."""
    from cli.parser import parse_args

    args = parse_args()
    if getattr(args, "command", None) == "doctor":
        from cli.doctor import run_doctor

        return run_doctor(json_output=bool(args.json))

    from main import main as application_main

    return application_main()
