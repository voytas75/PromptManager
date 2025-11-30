"""Unit tests for the Jinja2 template renderer and schema validation helpers.

Updates: v0.2.0 - 2025-11-27 - Assert syntax error hints for unmatched delimiters.
"""
from __future__ import annotations

import json

from core.templating import (
    SchemaValidationMode,
    SchemaValidator,
    TemplateRenderer,
    TemplateRenderResult,
)


def test_template_renderer_supports_custom_filters() -> None:
    """Custom filters should be usable inside the preview templates."""
    renderer = TemplateRenderer()
    template = "Hello {{ name | slugify }} -- {{ summary | truncate(5) }} -- {{ data | json }}"
    result = renderer.render(
        template,
        {
            "name": "Prompt Wizard",
            "summary": "abcdef",
            "data": {"a": 1},
        },
    )
    assert isinstance(result, TemplateRenderResult)
    assert result.errors == []
    assert result.rendered_text == "Hello prompt-wizard -- abcd… -- {\"a\": 1}"


def test_template_renderer_reports_missing_variables() -> None:
    """Undefined placeholders should be reported as missing variables."""
    renderer = TemplateRenderer()
    template = "{{ customer }} ordered {{ item }}"
    result = renderer.render(template, {"customer": "Ada"})
    assert result.rendered_text == ""
    assert "item" in result.missing_variables
    assert result.errors


def test_template_renderer_includes_hint_for_unmatched_braces() -> None:
    """Syntax errors should include helpful hints for unmatched delimiters."""
    renderer = TemplateRenderer()
    template = "{{ customer "
    result = renderer.render(template, {"customer": "Ada"})
    assert result.errors
    assert "Hint" in result.errors[0]
    assert "closing '}}'" in result.errors[0]


def test_schema_validator_rejects_invalid_json_payload() -> None:
    """Invalid JSON schemas are surfaced as structured validation errors."""
    validator = SchemaValidator()
    result = validator.validate(
        {"name": "Ada"},
        schema_text="not-json",
        mode=SchemaValidationMode.JSON_SCHEMA,
    )
    assert not result.is_valid
    assert result.schema_error is not None


def test_schema_validator_enforces_required_fields_json_schema() -> None:
    """JSON Schema validation flags missing required entries."""
    validator = SchemaValidator()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "minimum": 1},
        },
        "required": ["name", "count"],
    }
    result = validator.validate(
        {"name": "Ada"},
        schema_text=json.dumps(schema),
        mode=SchemaValidationMode.JSON_SCHEMA,
    )
    assert not result.is_valid
    assert any("count" in message for message in result.errors)


def test_schema_validator_enforces_required_fields_pydantic() -> None:
    """Pydantic conversion enforces schema requirements and typing."""
    validator = SchemaValidator()
    schema = {
        "title": "PreviewVariables",
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 3},
            "active": {"type": "boolean"},
        },
        "required": ["name"],
    }
    result = validator.validate(
        {"name": "Al"},
        schema_text=json.dumps(schema),
        mode=SchemaValidationMode.PYDANTIC,
    )
    assert not result.is_valid
    assert "name" in result.field_errors


def test_template_renderer_extracts_variables() -> None:
    """Variable extraction should ignore built-in placeholders."""
    renderer = TemplateRenderer()
    template = "{% for item in items %}{{ loop.index }} {{ item.name }}{% endfor %}"
    names = renderer.extract_variables(template)
    assert names == ["items"]
