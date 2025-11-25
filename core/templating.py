"""Jinja2 templating utilities for workspace previews and validation.

Updates: v0.1.0 - 2025-11-25 - Add strict Jinja2 renderer, custom filters,
and schema validation helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from jinja2 import Environment, StrictUndefined, Template, TemplateSyntaxError, UndefinedError, meta
from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions
from pydantic import BaseModel, Field, ValidationError, create_model

from models.category_model import slugify_category


def _truncate_filter(value: Any, limit: int = 500, suffix: str = "…") -> str:
    """Return ``value`` trimmed to ``limit`` characters with ``suffix`` appended."""

    text = str(value or "")
    if limit <= 0 or len(text) <= limit:
        return text
    if len(suffix) >= limit:
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def _slugify_filter(value: Any) -> str:
    """Return a slugified version of ``value`` using the project helper."""

    text = str(value or "")
    return slugify_category(text) or text.replace(" ", "-").lower()


def _json_filter(value: Any, *, indent: Optional[int] = None) -> str:
    """Return ``value`` serialized as JSON with optional pretty printing."""

    return json.dumps(value, ensure_ascii=False, indent=indent)


@dataclass(slots=True)
class TemplateRenderResult:
    """Outcome of rendering a template preview."""

    rendered_text: str
    errors: List[str] = field(default_factory=list)
    missing_variables: Set[str] = field(default_factory=set)


class TemplateRenderer:
    """Render Jinja2 templates with strict variable enforcement and custom filters."""

    _SPECIAL_VARIABLES: Set[str] = {"cycler", "loop", "namespace", "super", "caller"}

    def __init__(self) -> None:
        self._env = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters.update(
            {
                "truncate": _truncate_filter,
                "slugify": _slugify_filter,
                "json": _json_filter,
            }
        )

    def extract_variables(self, template_text: str) -> List[str]:
        """Return sorted placeholder names referenced within ``template_text``."""

        if not template_text.strip():
            return []
        parsed = self._env.parse(template_text)
        discovered = meta.find_undeclared_variables(parsed)
        return sorted(
            variable
            for variable in discovered
            if variable not in self._SPECIAL_VARIABLES
        )

    def render(self, template_text: str, variables: Mapping[str, Any]) -> TemplateRenderResult:
        """Render ``template_text`` with ``variables`` capturing syntax or undefined errors."""

        if not template_text.strip():
            return TemplateRenderResult(rendered_text="", errors=[])
        try:
            compiled: Template = self._env.from_string(template_text)
            rendered = compiled.render(**variables)
            return TemplateRenderResult(rendered_text=rendered, errors=[])
        except TemplateSyntaxError as exc:  # pragma: no cover - error path exercised in tests
            message = f"Syntax error on line {exc.lineno}: {exc.message}"
            return TemplateRenderResult(rendered_text="", errors=[message])
        except UndefinedError as exc:
            missing = self._infer_missing_variable(str(exc))
            return TemplateRenderResult(
                rendered_text="",
                errors=[str(exc)],
                missing_variables=missing,
            )

    @staticmethod
    def _infer_missing_variable(message: str) -> Set[str]:
        """Best-effort extraction of undefined variable names from Jinja errors."""

        pattern = re.compile(r"'(?P<name>[^']+)' is undefined")
        match = pattern.search(message)
        if match:
            return {match.group("name")}
        stripped = message.strip().strip("'")
        return {stripped} if stripped else set()


class SchemaValidationMode(Enum):
    """Schema validation strategies supported by the preview widget."""

    NONE = "none"
    JSON_SCHEMA = "json"
    PYDANTIC = "pydantic"

    @classmethod
    def from_string(cls, value: str | None) -> "SchemaValidationMode":
        if not value:
            return cls.NONE
        lowered = value.strip().lower()
        for member in cls:
            if member.value == lowered:
                return member
        return cls.NONE


@dataclass(slots=True)
class SchemaValidationResult:
    """Outcome of validating user variables against an optional schema."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    field_errors: Set[str] = field(default_factory=set)
    schema_error: Optional[str] = None


class SchemaValidator:
    """Validate variable dictionaries using JSON Schema or derived Pydantic models."""

    def validate(
        self,
        variables: Mapping[str, Any],
        schema_text: str,
        *,
        mode: SchemaValidationMode = SchemaValidationMode.NONE,
    ) -> SchemaValidationResult:
        """Validate ``variables`` against ``schema_text`` according to ``mode``."""

        if mode is SchemaValidationMode.NONE or not schema_text.strip():
            return SchemaValidationResult(is_valid=True)

        try:
            parsed_schema = json.loads(schema_text)
        except json.JSONDecodeError as exc:
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Invalid schema JSON: {exc.msg}"],
                schema_error=str(exc),
            )

        if not isinstance(parsed_schema, Mapping):
            return SchemaValidationResult(
                is_valid=False,
                errors=["Schema must be a JSON object."],
                schema_error="Schema root is not an object",
            )

        if mode is SchemaValidationMode.JSON_SCHEMA:
            return self._validate_with_json_schema(variables, parsed_schema)
        if mode is SchemaValidationMode.PYDANTIC:
            return self._validate_with_pydantic(variables, parsed_schema)
        return SchemaValidationResult(is_valid=True)

    def _validate_with_json_schema(
        self,
        variables: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> SchemaValidationResult:
        try:
            Draft202012Validator.check_schema(schema)
        except jsonschema_exceptions.SchemaError as exc:
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON Schema: {exc.message}"],
                schema_error=str(exc),
            )

        validator = Draft202012Validator(schema)
        errors: List[str] = []
        field_errors: Set[str] = set()
        for error in validator.iter_errors(dict(variables)):
            path = self._format_error_path(error.path)
            errors.append(f"{path}: {error.message}" if path else error.message)
            if path:
                field_errors.add(path)
        if errors:
            return SchemaValidationResult(is_valid=False, errors=errors, field_errors=field_errors)
        return SchemaValidationResult(is_valid=True)

    def _validate_with_pydantic(
        self,
        variables: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> SchemaValidationResult:
        try:
            model = self._create_model_from_schema(schema)
        except ValueError as exc:
            return SchemaValidationResult(
                is_valid=False,
                errors=[str(exc)],
                schema_error=str(exc),
            )

        try:
            model.model_validate(dict(variables))
        except ValidationError as exc:
            messages: List[str] = []
            field_errors: Set[str] = set()
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                messages.append(f"{location}: {error['msg']}")
                if location:
                    field_errors.add(location)
            return SchemaValidationResult(
                is_valid=False,
                errors=messages,
                field_errors=field_errors,
            )
        return SchemaValidationResult(is_valid=True)

    def _create_model_from_schema(self, schema: Mapping[str, Any]) -> type[BaseModel]:
        schema_type = schema.get("type", "object")
        if schema_type not in ("object", None):
            raise ValueError("Only object schemas are supported for Pydantic validation")

        properties = schema.get("properties")
        if not isinstance(properties, Mapping) or not properties:
            raise ValueError("Schema must define object properties for Pydantic validation")

        required = set(schema.get("required", []))
        fields: Dict[str, Tuple[Any, Any]] = {}
        for name, definition in properties.items():
            if not isinstance(definition, Mapping):
                raise ValueError(f"Schema property '{name}' must be an object definition")
            python_type = self._resolve_python_type(definition)
            default = definition.get("default", ... if name in required else None)
            constraints = self._field_constraints(definition)
            field_info = Field(default, description=definition.get("description"), **constraints)
            fields[name] = (python_type, field_info)

        model_name = str(schema.get("title") or "PromptVariables")
        return create_model(model_name, **fields)

    @staticmethod
    def _resolve_python_type(definition: Mapping[str, Any]) -> Any:
        schema_type = definition.get("type", "string")
        if schema_type == "string":
            return str
        if schema_type == "integer":
            return int
        if schema_type == "number":
            return float
        if schema_type == "boolean":
            return bool
        if schema_type == "array":
            items = definition.get("items")
            item_type = (
                SchemaValidator._resolve_python_type(items)
                if isinstance(items, Mapping)
                else Any
            )
            return List[item_type]  # type: ignore[index]
        if schema_type == "object":
            return Dict[str, Any]
        return Any

    @staticmethod
    def _field_constraints(definition: Mapping[str, Any]) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}
        if isinstance(definition.get("minLength"), int):
            constraints["min_length"] = definition["minLength"]
        if isinstance(definition.get("maxLength"), int):
            constraints["max_length"] = definition["maxLength"]
        if isinstance(definition.get("minimum"), (int, float)):
            constraints["ge"] = definition["minimum"]
        if isinstance(definition.get("maximum"), (int, float)):
            constraints["le"] = definition["maximum"]
        return constraints

    @staticmethod
    def _format_error_path(path: Sequence[Any]) -> str:
        return ".".join(str(part) for part in path)


__all__ = [
    "TemplateRenderer",
    "TemplateRenderResult",
    "SchemaValidator",
    "SchemaValidationResult",
    "SchemaValidationMode",
]
