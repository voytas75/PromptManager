"""Provider-free comparisons of current PromptManager prompt assets."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from core.templating import TemplateRenderer

if TYPE_CHECKING:
    from datetime import datetime

    from models.prompt_model import Prompt


_METADATA_FIELDS: tuple[str, ...] = (
    "description",
    "category",
    "category_slug",
    "tags",
    "language",
    "scenarios",
    "author",
    "related_prompts",
    "is_favorite",
)


@dataclass(frozen=True, slots=True)
class PromptComparisonReport:
    """Stable, provider-free comparison data for two current prompt assets."""

    left: Prompt
    right: Prompt
    state: dict[str, dict[str, object]]
    metadata_differences: dict[str, dict[str, object]]
    variables: dict[str, object]
    relationship: str
    operational_counters: dict[str, dict[str, int | float | None]]
    body_diff: str

    def to_record(self) -> dict[str, object]:
        """Return a JSON-ready comparison report."""
        return {
            "left": {"id": str(self.left.id), "name": self.left.name},
            "right": {"id": str(self.right.id), "name": self.right.name},
            "state": self.state,
            "metadata_differences": self.metadata_differences,
            "variables": self.variables,
            "lineage": {"relationship": self.relationship},
            "operational_counters": self.operational_counters,
            "body_diff": self.body_diff,
        }


def compare_prompts(
    left: Prompt,
    right: Prompt,
    *,
    left_parent_id: str | None = None,
    right_parent_id: str | None = None,
    renderer: TemplateRenderer | None = None,
) -> PromptComparisonReport:
    """Compare two current prompts without rendering, providers, or persistence access."""
    template_renderer = renderer or TemplateRenderer()
    left_variables, left_error = _extract_variables(left.context, template_renderer)
    right_variables, right_error = _extract_variables(right.context, template_renderer)
    relationship = _direct_relationship(
        left_id=str(left.id),
        right_id=str(right.id),
        left_parent_id=left_parent_id,
        right_parent_id=right_parent_id,
    )
    return PromptComparisonReport(
        left=left,
        right=right,
        state={"left": _state_projection(left), "right": _state_projection(right)},
        metadata_differences=_metadata_differences(left, right),
        variables=_variable_report(left_variables, right_variables, left_error, right_error),
        relationship=relationship,
        operational_counters={
            "left": _operational_counters(left),
            "right": _operational_counters(right),
        },
        body_diff=_body_diff(left, right),
    )


def _state_projection(prompt: Prompt) -> dict[str, object]:
    return {
        "id": str(prompt.id),
        "name": prompt.name,
        "version": prompt.version,
        "source": prompt.source,
        "is_active": prompt.is_active,
        "last_modified": _isoformat(prompt.last_modified),
    }


def _metadata_differences(left: Prompt, right: Prompt) -> dict[str, dict[str, object]]:
    differences: dict[str, dict[str, object]] = {}
    for field in _METADATA_FIELDS:
        left_value = _metadata_value(left, field)
        right_value = _metadata_value(right, field)
        if left_value != right_value:
            differences[field] = {"left": left_value, "right": right_value}
    return differences


def _metadata_value(prompt: Prompt, field: str) -> object:
    value = getattr(prompt, field)
    if isinstance(value, list):
        tags = cast("list[str]", value)
        return list(tags)
    return value


def _extract_variables(
    context: str | None,
    renderer: TemplateRenderer,
) -> tuple[list[str], str | None]:
    body = str(context or "")
    if not body.strip():
        return [], None
    try:
        return renderer.extract_variables(body), None
    except Exception as exc:
        return [], f"Invalid template syntax: {exc}"


def _variable_report(
    left_variables: list[str],
    right_variables: list[str],
    left_error: str | None,
    right_error: str | None,
) -> dict[str, object]:
    left_set = set(left_variables)
    right_set = set(right_variables)
    errors: list[dict[str, str]] = []
    if left_error is not None:
        errors.append({"side": "left", "message": left_error})
    if right_error is not None:
        errors.append({"side": "right", "message": right_error})
    return {
        "left": sorted(left_set),
        "right": sorted(right_set),
        "shared": sorted(left_set & right_set),
        "left_only": sorted(left_set - right_set),
        "right_only": sorted(right_set - left_set),
        "errors": errors,
    }


def _direct_relationship(
    *,
    left_id: str,
    right_id: str,
    left_parent_id: str | None,
    right_parent_id: str | None,
) -> str:
    if left_parent_id == right_id:
        return "left_is_child_of_right"
    if right_parent_id == left_id:
        return "right_is_child_of_left"
    return "none"


def _operational_counters(prompt: Prompt) -> dict[str, int | float | None]:
    rating_count = int(prompt.rating_count)
    rating_sum = float(prompt.rating_sum)
    return {
        "usage_count": int(prompt.usage_count),
        "rating_count": rating_count,
        "rating_sum": rating_sum,
        "average_rating": rating_sum / rating_count if rating_count else None,
        "quality_score": prompt.quality_score,
    }


def _body_diff(left: Prompt, right: Prompt) -> str:
    return "\n".join(
        difflib.unified_diff(
            str(left.context or "").splitlines(),
            str(right.context or "").splitlines(),
            fromfile=f"{left.name} ({left.id})",
            tofile=f"{right.name} ({right.id})",
            lineterm="",
        )
    )


def _isoformat(value: datetime) -> str:
    return value.isoformat()


__all__ = ["PromptComparisonReport", "compare_prompts"]
