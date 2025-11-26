"""Additional PromptRepository branch coverage tests.

Updates: v0.1.0 - 2025-10-30 - Add error-path coverage for repository helpers.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.category_registry import CategoryRegistry
from core.repository import (
    PromptRepository,
    RepositoryError,
    RepositoryNotFoundError,
    _json_dumps,
    _json_loads_list,
    _json_loads_optional,
    _parse_optional_datetime,
)
from models.category_model import PromptCategory
from models.prompt_model import ExecutionStatus, Prompt, PromptExecution


def _make_prompt(name: str = "Repo Branch Test") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="demo",
        category="tests",
    )


def test_json_helpers_cover_edge_cases() -> None:
    assert _json_dumps(None) is None
    assert _json_dumps(["a"]).startswith("[")

    assert _json_loads_list(None) == []
    assert _json_loads_list("null") == []
    assert _json_loads_list('["x",1]') == ["x", "1"]
    assert _json_loads_list("not-json") == ["not-json"]

    assert _json_loads_optional(None) is None
    assert _json_loads_optional("null") is None
    assert _json_loads_optional('{"a": 1}') == {"a": 1}
    assert _json_loads_optional("{invalid") == "{invalid"


def test_repository_add_duplicate_raises_error(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)
    with pytest.raises(RepositoryError):
        repo.add(prompt)


def test_repository_update_missing_prompt(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    missing = _make_prompt()
    with pytest.raises(RepositoryNotFoundError):
        repo.update(missing)


def test_repository_delete_missing_prompt(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    with pytest.raises(RepositoryNotFoundError):
        repo.delete(uuid.uuid4())


def test_repository_list_with_limit(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompts = [_make_prompt(f"p-{idx}") for idx in range(3)]
    for prompt in prompts:
        repo.add(prompt)

    limited = repo.list(limit=1)
    assert len(limited) == 1
    assert limited[0].id == prompts[0].id


def test_repository_execution_roundtrip(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)

    execution = PromptExecution(
        id=uuid.uuid4(),
        prompt_id=prompt.id,
        request_text="print('hello')",
        response_text="Looks good!",
        status=ExecutionStatus.SUCCESS,
        metadata={"usage": {"prompt_tokens": 10}},
        rating=6.0,
    )

    repo.add_execution(execution)

    loaded = repo.get_execution(execution.id)
    assert loaded.response_text == execution.response_text
    assert loaded.request_text.startswith("print")
    assert loaded.metadata["usage"]["prompt_tokens"] == 10
    assert loaded.rating == pytest.approx(6.0)

    by_prompt = repo.list_executions_for_prompt(prompt.id)
    assert [entry.id for entry in by_prompt] == [execution.id]
    assert by_prompt[0].rating == pytest.approx(6.0)

    recent = repo.list_executions(limit=1)
    assert recent[0].id == execution.id


def test_repository_filtered_execution_query(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt_a = _make_prompt("Prompt A")
    prompt_b = _make_prompt("Prompt B")
    repo.add(prompt_a)
    repo.add(prompt_b)

    success_execution = PromptExecution(
        id=uuid.uuid4(),
        prompt_id=prompt_a.id,
        request_text="success",
        response_text="done",
        status=ExecutionStatus.SUCCESS,
        metadata={"note": "important"},
        rating=9,
    )
    failed_execution = PromptExecution(
        id=uuid.uuid4(),
        prompt_id=prompt_b.id,
        request_text="failure",
        response_text="oops",
        status=ExecutionStatus.FAILED,
    )
    repo.add_execution(success_execution)
    repo.add_execution(failed_execution)

    filtered = repo.list_executions_filtered(status=ExecutionStatus.SUCCESS.value)
    assert len(filtered) == 1
    assert filtered[0].id == success_execution.id

    prompt_filtered = repo.list_executions_filtered(prompt_id=prompt_b.id)
    assert len(prompt_filtered) == 1
    assert prompt_filtered[0].id == failed_execution.id

    search_filtered = repo.list_executions_filtered(search="important")
    assert len(search_filtered) == 1
    assert search_filtered[0].id == success_execution.id


def test_prompt_version_roundtrip(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt("Versioned Prompt")
    repo.add(prompt)

    version_one = repo.record_prompt_version(prompt, commit_message="initial import")
    assert version_one.version_number == 1
    assert version_one.commit_message == "initial import"

    prompt.description = "Updated description"
    repo.update(prompt)
    version_two = repo.record_prompt_version(prompt)

    versions = repo.list_prompt_versions(prompt.id)
    assert [entry.version_number for entry in versions][:2] == [2, 1]

    fetched = repo.get_prompt_version(version_two.id)
    assert fetched.to_prompt().description == "Updated description"

    latest = repo.get_prompt_latest_version(prompt.id)
    assert latest is not None and latest.id == version_two.id


def test_prompt_fork_links(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    source = _make_prompt("Source")
    child = _make_prompt("Child")
    repo.add(source)
    repo.add(child)

    link = repo.record_prompt_fork(source.id, child.id)
    assert link.source_prompt_id == source.id

    parent = repo.get_prompt_parent_fork(child.id)
    assert parent is not None and parent.child_prompt_id == child.id

    children = repo.list_prompt_children(source.id)
    assert [entry.child_prompt_id for entry in children] == [child.id]


def test_category_update_propagates_prompt_label(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="tests", label="Tests", description="Testing prompts")
    repo.create_category(category)

    prompt = _make_prompt()
    prompt.category = category.label
    repo.add(prompt)

    updated_category = replace(
        category,
        label="Quality Checks",
        updated_at=datetime.now(timezone.utc),
    )
    repo.update_category(updated_category)

    refreshed = repo.get(prompt.id)
    assert refreshed.category == "Quality Checks"
    assert refreshed.category_slug == category.slug


def test_category_registry_ensure_creates_custom_entry(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    registry = CategoryRegistry(repo)
    custom = registry.ensure(slug="custom-support", label="Custom Support")
    assert custom.slug == "custom-support"
    stored = repo.list_categories(include_archived=True)
    assert any(entry.slug == "custom-support" for entry in stored)


def test_prompt_catalogue_stats_cover_edge_cases(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    recent = _make_prompt("Recent")
    recent.category = "Docs"
    recent.tags = ["docs"]
    stale = _make_prompt("Stale")
    stale.category = ""
    stale.tags = []
    stale.is_active = False
    stale.last_modified = datetime.now(timezone.utc) - timedelta(days=45)

    repo.add(recent)
    repo.add(stale)

    stats = repo.get_prompt_catalogue_stats()
    assert stats.total_prompts == 2
    assert stats.active_prompts == 1
    assert stats.inactive_prompts == 1
    assert stats.prompts_without_category == 1
    assert stats.prompts_without_tags == 1
    assert stats.distinct_categories == 1
    assert stats.distinct_tags == 1
    assert stats.stale_prompts == 1
    assert stats.average_tags_per_prompt == pytest.approx(0.5)
    assert isinstance(stats.last_modified_at, datetime)


def test_get_prompts_for_ids_preserves_order(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    first = _make_prompt("First")
    second = _make_prompt("Second")
    repo.add(first)
    repo.add(second)

    ordered = repo.get_prompts_for_ids([second.id, first.id, uuid.uuid4()])
    assert [prompt.id for prompt in ordered] == [second.id, first.id]


def test_set_category_active_toggles_state(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="docs", label="Docs", description="Docs prompts")
    repo.create_category(category)

    updated = repo.set_category_active("docs", False)
    assert updated.is_active is False

    with pytest.raises(RepositoryNotFoundError):
        repo.set_category_active("missing", True)


def test_parse_optional_datetime_variants() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    parsed_naive = _parse_optional_datetime(naive)
    assert parsed_naive.tzinfo is not None
    iso_text = "2024-01-02T03:04:05"
    parsed_text = _parse_optional_datetime(iso_text)
    assert parsed_text.tzinfo is not None
    assert _parse_optional_datetime("") is None


def test_sync_category_definitions_and_get_category(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    categories = [
        PromptCategory(slug="analysis", label="Analysis", description="Analysis prompts"),
        PromptCategory(slug="helpers", label="Helpers", description="Helper prompts"),
    ]
    created = repo.sync_category_definitions(categories)
    assert set(entry.slug for entry in created) == {"analysis", "helpers"}
    assert repo.get_category("analysis") is not None


def test_update_prompt_category_labels_applies_to_prompts(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="legacy", label="Legacy", description="Legacy")
    repo.create_category(category)
    prompt = _make_prompt("Needs relabel")
    prompt.category = "Legacy"
    prompt.category_slug = "legacy"
    repo.add(prompt)
    repo.update_prompt_category_labels("legacy", "Modernised")
    reloaded = repo.get(prompt.id)
    assert reloaded.category == "Modernised"


def test_update_execution_roundtrip(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt("Execution prompt")
    repo.add(prompt)
    execution = PromptExecution(
        id=uuid.uuid4(),
        prompt_id=prompt.id,
        request_text="inspect logs",
        response_text="done",
        status=ExecutionStatus.SUCCESS,
    )
    repo.add_execution(execution)
    execution.response_text = "updated"
    repo.update_execution(execution)
    assert repo.get_execution(execution.id).response_text == "updated"
    with pytest.raises(RepositoryNotFoundError):
        repo.update_execution(
            PromptExecution(
                id=uuid.uuid4(),
                prompt_id=prompt.id,
                request_text="missing",
                response_text="missing",
                status=ExecutionStatus.SUCCESS,
            )
        )


def test_create_category_duplicate_raises(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="dup", label="Duplicate", description="dup")
    repo.create_category(category)
    with pytest.raises(RepositoryError):
        repo.create_category(category)
