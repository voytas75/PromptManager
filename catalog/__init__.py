"""Built-in prompt catalog resources for Prompt Manager.

Updates: v0.5.0 - 2025-11-05 - Provide packaged default prompt catalogue.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any


def builtin_catalog_resource() -> Any:
    """Return a Traversable pointing to the packaged prompts JSON file."""
    return files(__name__).joinpath("prompts.json")


__all__ = ["builtin_catalog_resource"]
