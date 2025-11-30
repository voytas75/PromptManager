"""Settings management utilities for Prompt Manager configuration.

Updates:
  v0.5.2 - 2025-11-30 - Add category suggestion workflow to LiteLLM routing.
  v0.5.1 - 2025-11-23 - Added prompt template override settings.
  v0.5.0 - 2025-11-22 - Added structure-only prompt refinement routing.
  v0.4.9 - 2025-12-06 - Required LiteLLM-backed embeddings with UI fields.
  v0.4.8 - 2025-12-03 - Persisted chat colour palette overrides.
  v0.4.7 - 2025-11-05 - Added theme mode and chat appearance options.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

LITELLM_ROUTED_WORKFLOWS: OrderedDict[str, str] = OrderedDict(
    [
        ("name_generation", "Prompt name suggestions"),
        ("description_generation", "Prompt description synthesis"),
        ("scenario_generation", "Scenario drafting"),
        ("prompt_engineering", "Prompt refinement"),
        ("category_generation", "Prompt category suggestions"),
        ("prompt_structure_refinement", "Prompt structure refinement"),
        ("prompt_execution", "Prompt execution"),
    ]
)

LITELLM_ROUTING_OPTIONS: tuple[str, str] = ("fast", "inference")

DEFAULT_THEME_MODE = "light"
DEFAULT_CHAT_USER_BUBBLE_COLOR = "#e6f0ff"
DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR = "#f5f5f5"
DEFAULT_EMBEDDING_BACKEND = "litellm"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


class ChatColors(BaseSettings):
    """Sub-model storing UI colour customisation options."""

    user: str = Field(
        default=DEFAULT_CHAT_USER_BUBBLE_COLOR,
        description="User chat bubble colour (hex)",
    )
    assistant: str = Field(
        default=DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
        description="Assistant chat bubble colour (hex)",
    )

    model_config = cast(
        "SettingsConfigDict",
        {
            "env_prefix": "PROMPT_MANAGER_CHAT_",
            "case_sensitive": False,
            "env": {
                "user": ["CHAT_USER_BUBBLE_COLOR", "chat_user_bubble_color"],
                "assistant": ["CHAT_ASSISTANT_BUBBLE_COLOR", "chat_assistant_bubble_color"],
            },
        },
    )

_CHAT_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_THEME_CHOICES = {"light", "dark"}


class PromptTemplateOverrides(BaseModel):
    """User supplied overrides for the core LiteLLM system prompts."""

    name_generation: str | None = Field(
        default=None,
        description="System prompt text for the name generation workflow.",
    )
    description_generation: str | None = Field(
        default=None,
        description="System prompt text for the description generation workflow.",
    )
    scenario_generation: str | None = Field(
        default=None,
        description="System prompt text for the scenario generation workflow.",
    )
    prompt_engineering: str | None = Field(
        default=None,
        description="System prompt text for the prompt engineering workflow.",
    )
    category_generation: str | None = Field(
        default=None,
        description="System prompt text for the category suggestion workflow.",
    )


class SettingsError(Exception):
    """Raised when Prompt Manager configuration cannot be loaded or validated."""


class PromptManagerSettings(BaseSettings):
    """Application configuration sourced from environment variables or JSON files."""

    db_path: Path = Field(default=Path("data") / "prompt_manager.db")
    chroma_path: Path = Field(default=Path("data") / "chromadb")
    redis_dsn: str | None = None
    cache_ttl_seconds: int = Field(default=300)
    litellm_model: str | None = Field(
        default=None,
        description=(
            "LiteLLM fast model used for prompt naming, description hints, and other "
            "latency-sensitive workflows."
        ),
    )
    litellm_inference_model: str | None = Field(
        default=None,
        description=(
            "LiteLLM inference model for higher-quality, slower operations (configured "
            "separately from the fast model)."
        ),
    )
    litellm_api_key: str | None = Field(
        default=None,
        description="LiteLLM API key.",
        repr=False,
    )
    litellm_api_base: str | None = Field(
        default=None,
        description="Optional LiteLLM API base URL override.",
    )
    litellm_api_version: str | None = Field(
        default=None,
        description="Optional LiteLLM API version (useful for Azure OpenAI).",
    )
    litellm_drop_params: list[str] | None = Field(
        default=None,
        description=(
            "Optional LiteLLM parameters to drop before forwarding requests (see "
            "https://docs.litellm.ai/docs/completion/drop_params)."
        ),
    )
    litellm_reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Optional reasoning effort level for OpenAI reasoning models (minimal, "
            "medium, high)."
        ),
    )
    litellm_stream: bool = Field(
        default=False,
        description="Enable streaming responses when executing prompts via LiteLLM.",
    )
    litellm_workflow_models: dict[str, Literal["fast", "inference"]] | None = Field(
        default=None,
        description=(
            "Workflow-specific LiteLLM routing overrides mapping identifiers to "
            "'fast' or 'inference'."
        ),
    )
    quick_actions: list[dict[str, object]] | None = Field(
        default=None,
        description="Optional list of custom quick action definitions for the command palette.",
    )
    categories_path: Path | None = Field(
        default=None,
        description="Optional JSON file containing additional prompt category definitions.",
    )
    categories: list[dict[str, object]] | None = Field(
        default=None,
        description=(
            "Inline list of prompt category definitions sourced from environment "
            "variables or JSON config."
        ),
    )
    theme_mode: Literal["light", "dark"] = Field(
        default=DEFAULT_THEME_MODE,
        description="Preferred UI theme mode (light or dark).",
    )
    chat_user_bubble_color: str = Field(
        default=DEFAULT_CHAT_USER_BUBBLE_COLOR,
        description=(
            "Hex colour used when rendering user chat messages within the transcript tab."
        ),
    )

    chat_colors: ChatColors = Field(
        default_factory=ChatColors,
        description="Colour palette for chat bubbles.",
    )
    embedding_backend: str = Field(
        default=DEFAULT_EMBEDDING_BACKEND,
        description="Embedding backend to use (deterministic, LiteLLM, sentence-transformers).",
    )
    embedding_model: str | None = Field(
        default=None,
        description=(
            "Model name for the embedding backend (required for LiteLLM and "
            "sentence-transformers)."
        ),
    )
    embedding_device: str | None = Field(
        default=None,
        description="Preferred device identifier for local embedding backends (e.g. cpu, cuda).",
    )
    prompt_templates: PromptTemplateOverrides = Field(
        default_factory=PromptTemplateOverrides,
        description="Optional overrides for the built-in LiteLLM system prompts.",
    )

    # Pydantic v2 settings configuration
    model_config = cast(
        "SettingsConfigDict",
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
                "litellm_inference_model": ["LITELLM_INFERENCE_MODEL", "litellm_inference_model"],
                "litellm_api_key": ["LITELLM_API_KEY", "litellm_api_key", "AZURE_OPENAI_API_KEY"],
                "litellm_api_base": [
                    "LITELLM_API_BASE",
                    "litellm_api_base",
                    "AZURE_OPENAI_ENDPOINT",
                ],
                "litellm_api_version": [
                    "LITELLM_API_VERSION",
                    "litellm_api_version",
                    "AZURE_OPENAI_API_VERSION",
                ],
                "litellm_drop_params": [
                    "LITELLM_DROP_PARAMS",
                    "litellm_drop_params",
                ],
                "litellm_reasoning_effort": [
                    "LITELLM_REASONING_EFFORT",
                    "litellm_reasoning_effort",
                ],
                "litellm_stream": ["LITELLM_STREAM", "litellm_stream"],
                "litellm_workflow_models": ["LITELLM_WORKFLOW_MODELS", "litellm_workflow_models"],
                "embedding_backend": ["EMBEDDING_BACKEND", "embedding_backend"],
                "embedding_model": ["EMBEDDING_MODEL", "embedding_model"],
                "embedding_device": ["EMBEDDING_DEVICE", "embedding_device"],
                "quick_actions": ["QUICK_ACTIONS", "quick_actions"],
                "categories_path": ["CATEGORIES_PATH", "categories_path"],
                "categories": ["CATEGORIES", "categories"],
                "theme_mode": ["THEME_MODE", "theme_mode"],
                "chat_user_bubble_color": ["CHAT_USER_BUBBLE_COLOR", "chat_user_bubble_color"],
                "prompt_templates": ["PROMPT_TEMPLATES", "prompt_templates"],
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
    def _trim_redis_dsn(cls, value: str | None) -> str | None:
        """Normalise Redis DSN values by stripping whitespace and empty strings."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("categories_path", mode="before")
    def _normalise_categories_path(cls, value: Any) -> Path | None:
        """Coerce optional category file path into a Path."""

        if value in (None, ""):
            return None
        path = Path(str(value)).expanduser()
        return path.resolve()

    @field_validator("categories", mode="before")
    def _parse_categories(cls, value: Any) -> list[dict[str, object]] | None:
        """Ensure inline categories are represented as a list of mappings."""

        if value in (None, "", []):
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("categories must be valid JSON") from exc
            value = parsed
        if isinstance(value, list):
            cleaned: list[dict[str, object]] = []
            entries = cast("Sequence[object]", value)
            for entry in entries:
                if isinstance(entry, Mapping):
                    entry_mapping = cast("Mapping[object, object]", entry)
                    cleaned.append({str(key): entry_mapping[key] for key in entry_mapping})
            return cleaned or None
        raise ValueError("categories must be provided as a list of objects")


    @field_validator(
        "litellm_model",
        "litellm_inference_model",
        "litellm_api_key",
        "litellm_api_base",
        "embedding_model",
        "embedding_device",
        mode="before",
    )
    def _strip_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("embedding_backend", mode="before")
    def _normalise_embedding_backend(cls, value: str | None) -> str:
        if value is None:
            return DEFAULT_EMBEDDING_BACKEND
        backend = str(value).strip().lower()
        if backend in {"", "default", "deterministic"}:
            return "deterministic"
        if backend in {"litellm", "openai"}:
            return "litellm"
        if backend in {"sentence-transformers", "sentence_transformers", "st"}:
            return "sentence-transformers"
        raise ValueError(f"Unsupported embedding backend '{value}'")

    @field_validator("theme_mode", mode="before")
    def _normalise_theme_mode(cls, value: str | None) -> str:
        if value is None:
            return DEFAULT_THEME_MODE
        text = str(value).strip().lower()
        if text not in _THEME_CHOICES:
            return DEFAULT_THEME_MODE
        return text

    @field_validator("chat_user_bubble_color", mode="before")
    def _normalise_chat_colour(cls, value: str | None) -> str:
        if value is None:
            return DEFAULT_CHAT_USER_BUBBLE_COLOR
        text = str(value).strip()
        if not text:
            return DEFAULT_CHAT_USER_BUBBLE_COLOR
        match = _CHAT_COLOR_PATTERN.fullmatch(text)
        if match is None:
            raise ValueError("chat_user_bubble_color must be a hex colour such as #e6f0ff")
        if len(text) == 4:
            r, g, b = text[1], text[2], text[3]
            text = f"#{r}{r}{g}{g}{b}{b}"
        return text.lower()

    @model_validator(mode="after")
    def _validate_embedding_configuration(self) -> PromptManagerSettings:
        backend = self.embedding_backend
        model = self.embedding_model

        if backend == "litellm":
            resolved_model = model or DEFAULT_EMBEDDING_MODEL
            object.__setattr__(self, "embedding_model", resolved_model)
            model = resolved_model

        if backend == "deterministic" and model:
            object.__setattr__(self, "embedding_model", None)
            return self

        if backend != "deterministic" and not model:
            raise ValueError(
                "embedding_model must be provided when embedding_backend is set to "
                f"'{backend}'"
            )
        return self

    @field_validator("quick_actions", mode="before")
    def _validate_quick_actions(cls, value: object) -> list[dict[str, object]] | None:
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
        entries = cast("Sequence[object]", value)
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("quick_actions items must be objects")
            entry_mapping = cast("Mapping[object, object]", entry)
            normalised.append({str(key): entry_mapping[key] for key in entry_mapping})
        return normalised

    @field_validator("litellm_drop_params", mode="before")
    def _normalise_drop_params(cls, value: object) -> list[str] | None:
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
                if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
                    sequence = cast("Sequence[object]", parsed)
                    items = [str(item).strip() for item in sequence if str(item).strip()]
                else:
                    items = [str(parsed).strip()]
            return items or None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sequence_value = cast("Sequence[object]", value)
            items = [str(item).strip() for item in sequence_value if str(item).strip()]
            return items or None
        raise ValueError(
            "litellm_drop_params must be a list, comma-separated string, or JSON array"
        )

    @field_validator("litellm_reasoning_effort", mode="before")
    def _normalise_reasoning_effort(cls, value: object) -> str | None:
        if value in (None, ""):
            return None
        effort = str(value).strip().lower()
        if effort not in {"minimal", "medium", "high"}:
            raise ValueError("litellm_reasoning_effort must be one of: minimal, medium, high")
        return effort

    @field_validator("litellm_workflow_models", mode="before")
    def _normalise_workflow_models(
        cls, value: object
    ) -> dict[str, Literal["fast", "inference"]] | None:
        if value in (None, "", {}, ()):  # type: ignore[comparison-overlap]
            return None
        if isinstance(value, str):
            payload = value.strip()
            if not payload:
                return None
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid configuration
                raise ValueError(
                    "litellm_workflow_models must be a JSON object mapping workflow to model tier"
                ) from exc
        if not isinstance(value, Mapping):
            raise ValueError(
                "litellm_workflow_models must be a mapping of workflow names to model tiers"
            )
        mapping_value = cast("Mapping[object, object]", value)
        cleaned: dict[str, Literal["fast", "inference"]] = {}
        for raw_key, raw_value in mapping_value.items():
            key = str(raw_key).strip()
            if key not in LITELLM_ROUTED_WORKFLOWS:
                raise ValueError(f"Unsupported LiteLLM workflow '{key}'")
            if raw_value is None:
                continue
            choice = str(raw_value).strip().lower()
            if choice not in LITELLM_ROUTING_OPTIONS:
                raise ValueError(
                    "Workflow routing must be set to 'fast' or 'inference' when provided"
                )
            if choice == "fast":
                continue
            cleaned[key] = cast("Literal['fast', 'inference']", choice)
        return cleaned or None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[
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
        def env_with_aliases(_: BaseSettings | None = None) -> dict[str, Any]:
            data: dict[str, Any] = {}
            config_dict = cast("dict[str, Any]", cls.model_config)
            prefix = str(config_dict.get("env_prefix", ""))
            # Collect both alias and field-name keys from environment
            mapping = {
                "db_path": ["DB_PATH", "DATABASE_PATH", "db_path", "database_path"],
                "chroma_path": ["CHROMA_PATH", "chroma_path"],
                "redis_dsn": ["REDIS_DSN", "redis_dsn"],
                "cache_ttl_seconds": ["CACHE_TTL_SECONDS", "cache_ttl_seconds"],
                "litellm_model": ["LITELLM_MODEL", "litellm_model"],
                "litellm_inference_model": [
                    "LITELLM_INFERENCE_MODEL",
                    "litellm_inference_model",
                ],
                "litellm_api_key": [
                    "LITELLM_API_KEY",
                    "litellm_api_key",
                    "AZURE_OPENAI_API_KEY",
                ],
                "litellm_api_base": [
                    "LITELLM_API_BASE",
                    "litellm_api_base",
                    "AZURE_OPENAI_ENDPOINT",
                ],
                "litellm_api_version": [
                    "LITELLM_API_VERSION",
                    "litellm_api_version",
                    "AZURE_OPENAI_API_VERSION",
                ],
                "litellm_workflow_models": [
                    "LITELLM_WORKFLOW_MODELS",
                    "litellm_workflow_models",
                ],
                "embedding_backend": ["EMBEDDING_BACKEND", "embedding_backend"],
                "embedding_model": ["EMBEDDING_MODEL", "embedding_model"],
                "embedding_device": ["EMBEDDING_DEVICE", "embedding_device"],
                "quick_actions": ["QUICK_ACTIONS", "quick_actions"],
                "litellm_reasoning_effort": [
                    "LITELLM_REASONING_EFFORT",
                    "litellm_reasoning_effort",
                ],
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
                        if not val.strip():
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
            cast("PydanticBaseSettingsSource", env_with_aliases),
            file_secret_settings,
        )

    @classmethod
    def _json_config_settings_source(
        cls,
        _: type[BaseSettings],
    ) -> PydanticBaseSettingsSource:
        """Return settings extracted from an optional JSON config file."""

        def _loader(_: BaseSettings | None = None) -> dict[str, Any]:
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
                except OSError as exc:  # pragma: no cover - filesystem failure is env-specific
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
                mapping_data = cast("Mapping[object, Any]", data)
                data_dict: dict[str, Any] = {str(key): value for key, value in mapping_data.items()}
                mapped: dict[str, Any] = {}
                if "database_path" in data_dict and "db_path" not in data_dict:
                    mapped["db_path"] = data_dict["database_path"]
                disallowed_secret_keys = {
                    "litellm_api_key",
                    "AZURE_OPENAI_API_KEY",
                }
                removed_secrets = [
                    key
                    for key in disallowed_secret_keys
                    if key in data_dict and data_dict.pop(key, None) is not None
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
                    "litellm_inference_model",
                    "litellm_api_base",
                    "litellm_api_version",
                    "litellm_drop_params",
                    "litellm_reasoning_effort",
                    "litellm_stream",
                    "litellm_workflow_models",
                    "embedding_backend",
                    "embedding_model",
                    "embedding_device",
                    "quick_actions",
                    "chat_user_bubble_color",
                    "chat_colors",
                    "theme_mode",
                    "prompt_templates",
                ):
                    if key in data_dict:
                        mapped[key] = data_dict[key]
                if "litellm_api_base" not in mapped and "AZURE_OPENAI_ENDPOINT" in data_dict:
                    mapped["litellm_api_base"] = data_dict["AZURE_OPENAI_ENDPOINT"]
                if "litellm_api_version" not in mapped and "AZURE_OPENAI_API_VERSION" in data_dict:
                    mapped["litellm_api_version"] = data_dict["AZURE_OPENAI_API_VERSION"]
                return mapped
            return {}
        return cast("PydanticBaseSettingsSource", _loader)


def load_settings(**overrides: Any) -> PromptManagerSettings:
    """Return validated settings, raising SettingsError on failure."""
    try:
        return PromptManagerSettings(**overrides)
    except SettingsError:
        raise
    except ValidationError as exc:
        raise SettingsError("Invalid Prompt Manager configuration") from exc
logger = logging.getLogger("prompt_manager.settings")
