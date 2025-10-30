from __future__ import annotations

"""Settings management for Prompt Manager configuration.

Updates: v0.3.2 - 2025-10-30 - Adjusted settings precedence docs and examples.
Updates: v0.3.1 - 2025-10-30 - Fix settings source precedence so env overrides JSON; remove dead code.
Updates: v0.3.0 - 2025-10-30 - Migrate to Pydantic v2 with pydantic-settings.
Updates: v0.2.0 - 2025-11-03 - Introduced environment/JSON-backed settings with validation.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, ValidationError
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsError(Exception):
    """Raised when Prompt Manager configuration cannot be loaded or validated."""


class PromptManagerSettings(BaseSettings):
    """Application configuration sourced from environment variables or JSON files."""

    db_path: Path = Field(default=Path("data") / "prompt_manager.db", alias="database_path")
    chroma_path: Path = Field(default=Path("data") / "chromadb")
    redis_dsn: Optional[str] = None
    cache_ttl_seconds: int = Field(default=300)

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_prefix="PROMPT_MANAGER_",
        case_sensitive=False,
        populate_by_name=True,
        env={
            "db_path": ["DATABASE_PATH", "DB_PATH"],
            "chroma_path": ["CHROMA_PATH"],
            "redis_dsn": ["REDIS_DSN"],
            "cache_ttl_seconds": ["CACHE_TTL_SECONDS"],
        },
    )

    @field_validator("db_path", "chroma_path", mode="before")
    def _normalise_path(cls, value: Any) -> Path:
        """Expand user-relative paths and coerce values to Path instances."""
        if value is None:
            raise ValueError("a filesystem path is required")
        path = Path(str(value)).expanduser()
        return path.resolve()

    @field_validator("cache_ttl_seconds")
    def _validate_cache_ttl(cls, value: int) -> int:
        """Ensure the cache TTL is a positive integer."""
        if value <= 0:
            raise ValueError("cache_ttl_seconds must be greater than zero")
        return value

    @field_validator("redis_dsn", mode="before")
    def _trim_redis_dsn(cls, value: Optional[str]) -> Optional[str]:
        """Normalise Redis DSN values by stripping whitespace and empty strings."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        """Inject JSON configuration loading between init arguments and environment.

        Pydantic v2 uses `settings_customise_sources` with a different signature.
        """
        # Source precedence: init -> environment -> JSON -> file secrets.
        # Place `env_settings` before JSON so env overrides values from file.
        return (
            init_settings,
            env_settings,
            cls._json_config_settings_source(settings_cls),  # JSON file loader
            file_secret_settings,
        )

    @classmethod
    def _json_config_settings_source(cls, _: type[BaseSettings]):
        """Return settings extracted from an optional JSON config file."""
        def _loader() -> Dict[str, Any]:
            config_path = os.getenv("PROMPT_MANAGER_CONFIG_JSON")
            if not config_path:
                return {}
            resolved_path = Path(config_path).expanduser()
            if not resolved_path.exists():
                raise SettingsError(f"Configuration file not found: {resolved_path}")
            try:
                raw_contents = resolved_path.read_text(encoding="utf-8")
            except OSError as exc:  # pragma: no cover - filesystem failure is environment-specific
                raise SettingsError(f"Unable to read configuration file: {resolved_path}") from exc
            try:
                data = json.loads(raw_contents)
            except json.JSONDecodeError as exc:
                raise SettingsError(f"Invalid JSON in configuration file: {resolved_path}") from exc
            if not isinstance(data, dict):
                raise SettingsError(f"Configuration file {resolved_path} must contain a JSON object")
            return data
        return _loader


def load_settings(**overrides: Any) -> PromptManagerSettings:
    """Return validated settings, raising SettingsError on failure."""
    try:
        return PromptManagerSettings(**overrides)
    except SettingsError:
        raise
    except ValidationError as exc:
        raise SettingsError("Invalid Prompt Manager configuration") from exc
