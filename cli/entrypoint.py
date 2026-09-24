"""Lightweight console-script entrypoint for provider-free doctor startup."""

from __future__ import annotations


def main() -> int:
    """Dispatch doctor before importing the provider-backed application runtime."""
    from cli.parser import parse_args

    args = parse_args()
    if getattr(args, "command", None) == "doctor":
        from cli.doctor import run_doctor

        return run_doctor(
            json_output=bool(
                args.json
                or getattr(args, "catalog_json", False)
                or getattr(args, "doctor_json", False)
            ),
            command=args.doctor_command,
            details=bool(getattr(args, "details", False)),
            export_csv=getattr(args, "export_csv", None),
            action=getattr(args, "doctor_action", None),
            reference=getattr(args, "reference", None),
            suite=getattr(args, "suite", None),
            path=getattr(args, "path", None),
        )

    from main import main as application_main

    return application_main()
