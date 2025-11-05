"""Settings management for Prompt Manager configuration.

Updates: v0.4.3 - 2025-11-26 - Add LiteLLM streaming configuration flag.
Updates: v0.4.2 - 2025-11-15 - Ignore LiteLLM API secrets in JSON configuration files with warnings.
Updates: v0.4.1 - 2025-11-14 - Load LiteLLM API key from JSON configuration files.
Updates: v0.4.0 - 2025-11-07 - Add configurable embedding backend options.
Updates: v0.3.2 - 2025-10-30 - Document revised settings precedence examples.
Updates: v0.3.1 - 2025-10-30 - Ensure env overrides JSON; remove unused helpers.
Updates: v0.3.0 - 2025-10-30 - Migrate to Pydantic v2 with pydantic-settings.
Updates: v0.2.0 - 2025-11-03 - Introduce env/JSON-backed settings with validation.
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from pydantic import Field, ValidationError, field_validator, model_validator
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
    litellm_model: Optional[str] = Field(default=None, description="LiteLLM model name for prompt name generation.")
    litellm_api_key: Optional[str] = Field(default=None, description="LiteLLM API key.", repr=False)
    litellm_api_base: Optional[str] = Field(default=None, description="Optional LiteLLM API base URL override.")
    litellm_api_version: Optional[str] = Field(
        default=None,
        description="Optional LiteLLM API version (useful for Azure OpenAI).",
    )
    litellm_drop_params: Optional[List[str]] = Field(
        default=None,
        description="Optional list of LiteLLM parameters to drop before forwarding requests (see https://docs.litellm.ai/docs/completion/drop_params).",
    )
    litellm_reasoning_effort: Optional[str] = Field(
        default=None,
        description="Optional reasoning effort level for OpenAI reasoning models (minimal, medium, high).",
    )
    litellm_stream: bool = Field(
        default=False,
        description="Enable streaming responses when executing prompts via LiteLLM.",
    )
    quick_actions: Optional[list[dict[str, object]]] = Field(
        default=None,
        description="Optional list of custom quick action definitions for the command palette.",
    )
    embedding_backend: str = Field(
        default="deterministic",
        description="Embedding backend to use (deterministic, litellm, sentence-transformers).",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Model name for embedding backend (required for litellm/sentence-transformers).",
    )
    embedding_device: Optional[str] = Field(
        default=None,
        description="Preferred device identifier for local embedding backends (e.g. cpu, cuda).",
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
                "litellm_model": ["LITELLM_MODEL", "litellm_model"],
                "litellm_api_key": ["LITELLM_API_KEY", "litellm_api_key", "AZURE_OPENAI_API_KEY"],
                "litellm_api_base": ["LITELLM_API_BASE", "litellm_api_base", "AZURE_OPENAI_ENDPOINT"],
                "litellm_api_version": ["LITELLM_API_VERSION", "litellm_api_version", "AZURE_OPENAI_API_VERSION"],
                "litellm_drop_params": ["LITELLM_DROP_PARAMS", "litellm_drop_params"],
                "litellm_reasoning_effort": ["LITELLM_REASONING_EFFORT", "litellm_reasoning_effort"],
                "litellm_stream": ["LITELLM_STREAM", "litellm_stream"],
                "embedding_backend": ["EMBEDDING_BACKEND", "embedding_backend"],
                "embedding_model": ["EMBEDDING_MODEL", "embedding_model"],
                "embedding_device": ["EMBEDDING_DEVICE", "embedding_device"],
                "quick_actions": ["QUICK_ACTIONS", "quick_actions"],
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


    @field_validator(
        "litellm_model",
        "litellm_api_key",
        "litellm_api_base",
        "embedding_model",
        "embedding_device",
        mode="before",
    )
    def _strip_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("embedding_backend", mode="before")
    def _normalise_embedding_backend(cls, value: Optional[str]) -> str:
        if value is None:
            return "deterministic"
        backend = str(value).strip().lower()
        if backend in {"", "default", "deterministic"}:
            return "deterministic"
        if backend in {"litellm", "openai"}:
            return "litellm"
        if backend in {"sentence-transformers", "sentence_transformers", "st"}:
            return "sentence-transformers"
        raise ValueError(f"Unsupported embedding backend '{value}'")

    @model_validator(mode="after")
    def _validate_embedding_configuration(self) -> "PromptManagerSettings":
        backend = self.embedding_backend
        model = self.embedding_model

        if model and backend == "deterministic":
            backend = "litellm"
            object.__setattr__(self, "embedding_backend", backend)

        if backend == "litellm" and not model and self.litellm_model:
            model = self.litellm_model
            object.__setattr__(self, "embedding_model", model)

        if backend != "deterministic" and not model:
            raise ValueError(
                "embedding_model must be provided when embedding_backend is set to "
                f"'{backend}'"
            )
        return self

    @field_validator("quick_actions", mode="before")
    def _validate_quick_actions(cls, value: object) -> Optional[list[dict[str, object]]]:
        if value in (None, "", []):
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("quick_actions must be valid JSON") from exc
            value = parsed
        if not isinstance(value, list):
            raise ValueError("quick_actions must be a list of action definitions")
        normalised: list[dict[str, object]] = []
        for entry in value:
            if not isinstance(entry, dict):
                raise ValueError("quick_actions items must be objects")
            normalised.append({str(key): entry[key] for key in entry})
        return normalised

    @field_validator("litellm_drop_params", mode="before")
    def _normalise_drop_params(cls, value: object) -> Optional[List[str]]:
        if value in (None, "", [], ()):  # type: ignore[comparison-overlap]
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                items = [item.strip() for item in stripped.split(",") if item.strip()]
            else:
                if isinstance(parsed, list):
                    items = [str(item).strip() for item in parsed if str(item).strip()]
                elif isinstance(parsed, (tuple, set)):
                    items = [str(item).strip() for item in parsed if str(item).strip()]
                else:
                    items = [str(parsed).strip()]
            return items or None
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if str(item).strip()]
            return items or None
        raise ValueError("litellm_drop_params must be a list, comma-separated string, or JSON array")

    @field_validator("litellm_reasoning_effort", mode="before")
    def _normalise_reasoning_effort(cls, value: object) -> Optional[str]:
        if value in (None, ""):
            return None
        effort = str(value).strip().lower()
        if effort not in {"minimal", "medium", "high"}:
            raise ValueError("litellm_reasoning_effort must be one of: minimal, medium, high")
        return effort

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
        """Define configuration source precedence.

        Order (highest → lowest):
            1. Explicit keyword arguments (e.g. load_settings(litellm_model="...")).
            2. JSON configuration file (application settings).
            3. Environment variables / aliases.
            4. File secrets.
        """
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
                "litellm_model": ["LITELLM_MODEL", "litellm_model"],
                "litellm_api_key": ["LITELLM_API_KEY", "litellm_api_key", "AZURE_OPENAI_API_KEY"],
                "litellm_api_base": ["LITELLM_API_BASE", "litellm_api_base", "AZURE_OPENAI_ENDPOINT"],
                "litellm_api_version": ["LITELLM_API_VERSION", "litellm_api_version", "AZURE_OPENAI_API_VERSION"],
                "embedding_backend": ["EMBEDDING_BACKEND", "embedding_backend"],
                "embedding_model": ["EMBEDDING_MODEL", "embedding_model"],
                "embedding_device": ["EMBEDDING_DEVICE", "embedding_device"],
                "quick_actions": ["QUICK_ACTIONS", "quick_actions"],
                "litellm_reasoning_effort": ["LITELLM_REASONING_EFFORT", "litellm_reasoning_effort"],
                "litellm_stream": ["LITELLM_STREAM", "litellm_stream"],
            }
            for field, keys in mapping.items():
                for key in keys:
                    val = os.getenv(f"{prefix}{key}")
                    if val is None:
                        val = os.getenv(f"{prefix}{key.upper()}")
                    if val is None and key.isupper():
                        val = os.getenv(key)
                    if val is not None:
                        if isinstance(val, str) and not val.strip():
                            continue
                        # Map to canonical field keys accepted by the model
                        if field in {"db_path", "chroma_path"}:
                            data[field] = str(val)
                        else:
                            data[field] = val
                        break
            return data

        return (
            init_settings,
            cls._json_config_settings_source(settings_cls),
            cast(PydanticBaseSettingsSource, env_with_aliases),
            file_secret_settings,
        )

    @classmethod
    def _json_config_settings_source(
        cls,
        _: Type[BaseSettings],
    ) -> PydanticBaseSettingsSource:
        """Return settings extracted from an optional JSON config file."""

        def _loader(_: BaseSettings | None = None) -> Dict[str, Any]:
            explicit_path = os.getenv("PROMPT_MANAGER_CONFIG_JSON")
            candidates: list[Path] = []
            if explicit_path:
                candidates.append(Path(explicit_path).expanduser())
            candidates.append((Path("config") / "config.json").expanduser())

            for path in candidates:
                if not path:
                    continue
                if not path.exists():
                    if path == candidates[0]:
                        raise SettingsError(f"Configuration file not found: {path}")
                    continue
                try:
                    raw_contents = path.read_text(encoding="utf-8")
                except OSError as exc:  # pragma: no cover - filesystem failure is environment-specific
                    raise SettingsError(f"Unable to read configuration file: {path}") from exc
                try:
                    data = json.loads(raw_contents)
                except json.JSONDecodeError as exc:
                    raise SettingsError(
                        f"Invalid JSON in configuration file: {path}"
                    ) from exc
                if not isinstance(data, dict):
                    message = f"Configuration file {path} must contain a JSON object"
                    raise SettingsError(message)
                mapped: Dict[str, Any] = {}
                if "database_path" in data and "db_path" not in data:
                    mapped["db_path"] = data["database_path"]
                disallowed_secret_keys = {
                    "litellm_api_key",
                    "AZURE_OPENAI_API_KEY",
                }
                removed_secrets = [
                    key for key in disallowed_secret_keys if key in data and data.pop(key, None) is not None
                ]
                if removed_secrets:
                    logger.warning(
                        "Ignoring LiteLLM secret key(s) %s in configuration file %s; "
                        "set credentials via environment variables instead.",
                        ", ".join(sorted(removed_secrets)),
                        path,
                    )

                for key in (
                    "chroma_path",
                    "redis_dsn",
                    "cache_ttl_seconds",
                    "litellm_model",
                    "litellm_api_base",
                    "litellm_api_version",
                    "litellm_drop_params",
                    "litellm_reasoning_effort",
                    "litellm_stream",
                    "embedding_backend",
                    "embedding_model",
                    "embedding_device",
                    "quick_actions",
                ):
                    if key in data:
                        mapped[key] = data[key]
                if "litellm_api_base" not in mapped and "AZURE_OPENAI_ENDPOINT" in data:
                    mapped["litellm_api_base"] = data["AZURE_OPENAI_ENDPOINT"]
                if "litellm_api_version" not in mapped and "AZURE_OPENAI_API_VERSION" in data:
                    mapped["litellm_api_version"] = data["AZURE_OPENAI_API_VERSION"]
                return mapped
            return {}
        return cast(PydanticBaseSettingsSource, _loader)


def load_settings(**overrides: Any) -> PromptManagerSettings:
    """Return validated settings, raising SettingsError on failure."""
    try:
        return PromptManagerSettings(**overrides)
    except SettingsError:
        raise
    except ValidationError as exc:
        raise SettingsError("Invalid Prompt Manager configuration") from exc
logger = logging.getLogger("prompt_manager.settings")
