"""Additional PromptRepository branch coverage tests.

Updates: v0.1.0 - 2025-10-30 - Add error-path coverage for repository helpers.
"""

from __future__ import annotations

import sqlite3
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
    _json_loads_dict,
    _json_loads_list,
    _json_loads_optional,
    _parse_optional_datetime,
)
from models.category_model import PromptCategory
from models.prompt_model import ExecutionStatus, Prompt, PromptExecution
from models.prompt_note import PromptNote
from models.response_style import ResponseStyle


def _make_prompt(name: str = "Repo Branch Test") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="demo",
        category="tests",
    )


def _make_response_style(name: str = "Detailed") -> ResponseStyle:
    return ResponseStyle(
        id=uuid.uuid4(),
        name=name,
        description="Detailed responses",
        tone="formal",
        voice="assistant",
        format_instructions="json",
        guidelines="Keep responses structured.",
        tags=["docs"],
        examples=["Example output"],
        metadata={"key": "value"},
        is_active=True,
        version="1.0",
    )


def _make_prompt_note(text: str = "Remember to review logs") -> PromptNote:
    return PromptNote(id=uuid.uuid4(), note=text)


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
    assert _json_loads_dict(None) == {}
    assert _json_loads_dict("not-json") == {}
    assert _json_loads_dict('{"a": 1}') == {"a": 1}


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


def test_repository_add_execution_duplicate_error(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)
    execution = PromptExecution(
        id=uuid.uuid4(),
        prompt_id=prompt.id,
        request_text="print('hello')",
        response_text="Looks good!",
        status=ExecutionStatus.SUCCESS,
    )
    repo.add_execution(execution)
    with pytest.raises(RepositoryError):
        repo.add_execution(execution)
    with pytest.raises(RepositoryNotFoundError):
        repo.get_execution(uuid.uuid4())


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

    ascending = repo.list_executions_filtered(order_desc=False, limit=1)
    assert ascending[0].id == success_execution.id


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
    with pytest.raises(RepositoryNotFoundError):
        repo.get_prompt_version(9999)


def test_prompt_version_listing_helpers(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt("Version Sequence")
    repo.add(prompt)
    first = repo.record_prompt_version(prompt, commit_message="first")
    prompt.description = "Second"
    repo.update(prompt)
    second = repo.record_prompt_version(prompt, commit_message="second")
    versions = repo.list_prompt_versions(prompt.id)
    assert [entry.id for entry in versions][:2] == [second.id, first.id]
    fetched = repo.get_prompt_version(second.id)
    assert fetched.commit_message == "second"


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
    assert repo.get_prompt_parent_fork(uuid.uuid4()) is None


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


def test_repository_backfills_category_slugs(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE prompts (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                category TEXT,
                tags TEXT,
                language TEXT,
                context TEXT,
                example_input TEXT,
                example_output TEXT,
                last_modified TEXT,
                version TEXT,
                author TEXT,
                quality_score REAL,
                usage_count INTEGER,
                rating_count INTEGER,
                rating_sum REAL,
                related_prompts TEXT,
                created_at TEXT,
                modified_by TEXT,
                is_active INTEGER,
                source TEXT,
                checksum TEXT,
                ext1 TEXT,
                ext2 TEXT,
                ext3 TEXT,
                ext4 TEXT,
                ext5 TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO prompts (
                id, name, description, category, tags, language, context, example_input,
                example_output, last_modified, version, author, quality_score, usage_count,
                rating_count, rating_sum, related_prompts, created_at, modified_by,
                is_active, source, checksum, ext1, ext2, ext3, ext4, ext5
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(uuid.uuid4()),
                "Legacy",
                "Legacy prompt",
                "Reporting",
                "[]",
                "en",
                "",
                "",
                "",
                datetime.now(timezone.utc).isoformat(),
                "1.0",
                None,
                None,
                0,
                0,
                0.0,
                "[]",
                datetime.now(timezone.utc).isoformat(),
                None,
                1,
                "local",
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
    repo = PromptRepository(str(db_path))
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(prompts);").fetchall()}
        assert "category_slug" in columns
        slug = conn.execute("SELECT category_slug FROM prompts;").fetchone()[0]
        assert slug == "reporting"


def test_list_categories_include_archived(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="docs", label="Docs", description="Docs")
    repo.create_category(category)
    repo.set_category_active("docs", False)
    active_only = repo.list_categories()
    assert all(cat.slug != "docs" for cat in active_only)
    archived = repo.list_categories(include_archived=True)
    assert any(cat.slug == "docs" for cat in archived)


def test_get_category_normalizes_slug(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="helpers", label="Helpers", description="Helpers")
    repo.create_category(category)
    found = repo.get_category("Helpers")
    assert found is not None and found.slug == "helpers"


def test_set_category_active_toggles_state(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="docs", label="Docs", description="Docs prompts")
    repo.create_category(category)

    updated = repo.set_category_active("docs", False)
    assert updated.is_active is False

    with pytest.raises(RepositoryNotFoundError):
        repo.set_category_active("missing", True)


def test_user_profile_roundtrip(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    profile = repo.get_user_profile()
    profile.username = "tester"
    repo.save_user_profile(profile)
    refreshed = repo.get_user_profile()
    assert refreshed.username == "tester"


def test_repository_update_category_sql_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    category = PromptCategory(slug="docs", label="Docs", description="Docs")
    repo.create_category(category)

    def fake_connect(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("fail")

    monkeypatch.setattr("core.repository.sqlite3.connect", fake_connect)
    with pytest.raises(RepositoryError):
        repo.update_category(category)


def test_response_style_crud(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    style = _make_response_style()
    repo.add_response_style(style)
    fetched = repo.get_response_style(style.id)
    assert fetched.name == style.name
    style.description = "Updated description"
    repo.update_response_style(style)
    style.is_active = False
    repo.update_response_style(style)
    assert all(entry.id != style.id for entry in repo.list_response_styles())
    all_styles = repo.list_response_styles(include_inactive=True)
    assert any(entry.id == style.id for entry in all_styles)
    repo.delete_response_style(style.id)
    with pytest.raises(RepositoryNotFoundError):
        repo.get_response_style(style.id)


def test_response_style_sql_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    style = _make_response_style()

    def fake_connect(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("boom")

    monkeypatch.setattr("core.repository.sqlite3.connect", fake_connect)
    with pytest.raises(RepositoryError):
        repo.add_response_style(style)


def test_prompt_note_crud(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    note = _make_prompt_note()
    repo.add_prompt_note(note)
    note.note = "Updated note"
    repo.update_prompt_note(note)
    notes = repo.list_prompt_notes()
    assert notes[0].note == "Updated note"
    repo.delete_prompt_note(note.id)
    with pytest.raises(RepositoryNotFoundError):
        repo.get_prompt_note(note.id)


def test_prompt_note_errors(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    note = _make_prompt_note()
    with pytest.raises(RepositoryNotFoundError):
        repo.update_prompt_note(note)
    with pytest.raises(RepositoryNotFoundError):
        repo.delete_prompt_note(note.id)


def test_parse_optional_datetime_variants() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    parsed_naive = _parse_optional_datetime(naive)
    assert parsed_naive.tzinfo is not None
    iso_text = "2024-01-02T03:04:05"
    parsed_text = _parse_optional_datetime(iso_text)
    assert parsed_text.tzinfo is not None
    assert _parse_optional_datetime("") is None
    assert _parse_optional_datetime("not-a-date") is None


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
