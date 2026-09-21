"""Tests for provider-free comparison of current prompt assets."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from core.prompt_comparison import compare_prompts
from models.prompt_model import Prompt


def _prompt(
    prompt_id: uuid.UUID,
    *,
    name: str,
    context: str,
    tags: list[str] | None = None,
    usage_count: int = 0,
    rating_count: int = 0,
    rating_sum: float = 0.0,
) -> Prompt:
    """Create a compact prompt asset for comparison checks."""
    return Prompt(
        id=prompt_id,
        name=name,
        description="Compare current prompt assets.",
        category="Engineering",
        tags=tags or [],
        context=context,
        usage_count=usage_count,
        rating_count=rating_count,
        rating_sum=rating_sum,
        last_modified=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
    )


def test_compare_prompts_reports_metadata_variables_body_and_direct_lineage() -> None:
    """Compare current assets without rendering templates or calling providers."""
    left_id = uuid.uuid4()
    right_id = uuid.uuid4()
    left = _prompt(
        left_id,
        name="Base triage",
        context="Review {{ repository }} for {{ failure }}.",
        tags=["ci"],
        usage_count=4,
        rating_count=2,
        rating_sum=9.0,
    )
    right = _prompt(
        right_id,
        name="Fork triage",
        context="Review {{ repository }} for {{ failure }} and {{ owner }}.",
        tags=["ci", "incident"],
        usage_count=1,
    )

    report = compare_prompts(right, left, left_parent_id=str(left_id))
    payload = cast("dict[str, Any]", report.to_record())

    assert payload["lineage"] == {"relationship": "left_is_child_of_right"}
    assert payload["metadata_differences"]["tags"] == {
        "left": ["ci", "incident"],
        "right": ["ci"],
    }
    assert payload["variables"] == {
        "left": ["failure", "owner", "repository"],
        "right": ["failure", "repository"],
        "shared": ["failure", "repository"],
        "left_only": ["owner"],
        "right_only": [],
        "errors": [],
    }
    assert payload["operational_counters"]["left"]["average_rating"] is None
    assert payload["operational_counters"]["right"]["average_rating"] == 4.5
    assert "--- Fork triage" in payload["body_diff"]
    assert "+Review {{ repository }} for {{ failure }}." in payload["body_diff"]


def test_compare_prompts_reports_identical_assets_without_differences() -> None:
    """Same prompt comparison gives a stable no-difference report."""
    prompt_id = uuid.uuid4()
    prompt = _prompt(prompt_id, name="Same", context="Review {{ repository }}.")

    payload = cast("dict[str, Any]", compare_prompts(prompt, prompt).to_record())

    assert payload["metadata_differences"] == {}
    assert payload["variables"]["shared"] == ["repository"]
    assert payload["variables"]["left_only"] == []
    assert payload["variables"]["right_only"] == []
    assert payload["lineage"] == {"relationship": "none"}
    assert payload["body_diff"] == ""


def test_compare_prompts_keeps_template_parse_errors_as_report_data() -> None:
    """Invalid stored templates are comparison data, not an exception."""
    left = _prompt(uuid.uuid4(), name="Broken", context="{{ unclosed ")
    right = _prompt(uuid.uuid4(), name="Plain", context="No variables.")

    payload = cast("dict[str, Any]", compare_prompts(left, right).to_record())

    assert payload["variables"]["left"] == []
    assert payload["variables"]["errors"][0]["side"] == "left"
    assert "Invalid template syntax:" in payload["variables"]["errors"][0]["message"]
