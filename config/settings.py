"""Settings management for Prompt Manager configuration.

Updates: v0.3.2 - 2025-10-30 - Document revised settings precedence examples.
Updates: v0.3.1 - 2025-10-30 - Ensure env overrides JSON; remove unused helpers.
Updates: v0.3.0 - 2025-10-30 - Migrate to Pydantic v2 with pydantic-settings.
Updates: v0.2.0 - 2025-11-03 - Introduce env/JSON-backed settings with validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, cast

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class SettingsError(Exception):
    """Raised when Prompt Manager configuration cannot be loaded or validated."""


class PromptManagerSettings(BaseSettings):
    """Application configuration sourced from environment variables or JSON files."""

    db_path: Path = Field(default=Path("data") / "prompt_manager.db")
    chroma_path: Path = Field(default=Path("data") / "chromadb")
    redis_dsn: Optional[str] = None
    cache_ttl_seconds: int = Field(default=300)
    catalog_path: Optional[Path] = Field(
        default=None,
        description="Optional path to a prompt catalogue (JSON file or directory).",
    )

    # Pydantic v2 settings configuration
    model_config = cast(
        SettingsConfigDict,
        {
            "env_prefix": "PROMPT_MANAGER_",
            "case_sensitive": False,
            "populate_by_name": True,
            # pydantic-settings v2 does not read env via field aliases by default.
            # Provide canonical field names alongside legacy aliases.
            "env": {
                # Accept legacy database path aliases alongside the canonical key.
                "db_path": ["DB_PATH", "DATABASE_PATH", "db_path", "database_path"],
                "chroma_path": ["CHROMA_PATH", "chroma_path"],
                "redis_dsn": ["REDIS_DSN", "redis_dsn"],
                "cache_ttl_seconds": ["CACHE_TTL_SECONDS", "cache_ttl_seconds"],
                "catalog_path": ["CATALOG_PATH", "catalog_path"],
            },
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

    @field_validator("catalog_path", mode="before")
    def _normalise_catalog_path(cls, value: Optional[Any]) -> Optional[Path]:
        """Normalise optional catalogue paths when provided."""
        if value in (None, "", False):
            return None
        path = Path(str(value)).expanduser()
        return path.resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
    ]:
        """Inject JSON configuration loading between init arguments and environment.

        Pydantic v2 uses `settings_customise_sources` with a different signature.
        """
        # Source precedence: init -> environment -> JSON -> file secrets.
        # Place `env_settings` before JSON so env overrides values from file.
        # Compose an environment source that also considers aliases explicitly
        def env_with_aliases(_: BaseSettings | None = None) -> Dict[str, Any]:
            data: Dict[str, Any] = {}
            config_dict = cast(Dict[str, Any], cls.model_config)
            prefix = str(config_dict.get("env_prefix", ""))
            # Collect both alias and field-name keys from environment
            mapping = {
                "db_path": ["DB_PATH", "DATABASE_PATH", "db_path", "database_path"],
                "chroma_path": ["CHROMA_PATH", "chroma_path"],
                "redis_dsn": ["REDIS_DSN", "redis_dsn"],
                "cache_ttl_seconds": ["CACHE_TTL_SECONDS", "cache_ttl_seconds"],
                "catalog_path": ["CATALOG_PATH", "catalog_path"],
            }
            for field, keys in mapping.items():
                for key in keys:
                    val = os.getenv(f"{prefix}{key}")
                    if val is not None:
                        # Map to canonical field keys accepted by the model
                        if field in {"db_path", "chroma_path"}:
                            data[field] = str(val)
                        else:
                            data[field] = val
                        break
            return data

        return (
            init_settings,
            cast(PydanticBaseSettingsSource, env_with_aliases),
            cls._json_config_settings_source(settings_cls),  # JSON file loader
            file_secret_settings,
        )

    @classmethod
    def _json_config_settings_source(
        cls,
        _: Type[BaseSettings],
    ) -> PydanticBaseSettingsSource:
        """Return settings extracted from an optional JSON config file."""

        def _loader(_: BaseSettings | None = None) -> Dict[str, Any]:
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
                raise SettingsError(
                    f"Invalid JSON in configuration file: {resolved_path}"
                ) from exc
            if not isinstance(data, dict):
                message = f"Configuration file {resolved_path} must contain a JSON object"
                raise SettingsError(message)
            # Map JSON alias keys to canonical field names for pydantic v2
            mapped: Dict[str, Any] = {}
            if "database_path" in data and "db_path" not in data:
                mapped["db_path"] = data["database_path"]
            for key in ("chroma_path", "redis_dsn", "cache_ttl_seconds", "catalog_path"):
                if key in data:
                    mapped[key] = data[key]
            return mapped
        return cast(PydanticBaseSettingsSource, _loader)


def load_settings(**overrides: Any) -> PromptManagerSettings:
    """Return validated settings, raising SettingsError on failure."""
    try:
        return PromptManagerSettings(**overrides)
    except SettingsError:
        raise
    except ValidationError as exc:
        raise SettingsError("Invalid Prompt Manager configuration") from exc
