"""Refuse prompt deletion when persisted relations still refer to its ID."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from typing import TYPE_CHECKING, Any, cast, override

import pytest
from chromadb.errors import ChromaError

from core.catalog_check import run_catalog_check
from core.exceptions import PromptDeletionBlockedError, PromptDeletionPartialError
from core.prompt_manager import PromptManager, PromptVersionError
from core.repository import PromptRepository, RepositoryError, RepositoryNotFoundError
from core.repository.prompt_dependencies import PromptDeleteBlockedError
from models.prompt_chain_model import PromptChain, PromptChainStep
from models.prompt_model import Prompt

if TYPE_CHECKING:
    from pathlib import Path


def _prompt(name: str, *, active: bool = True) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Synthetic",
        category="Test",
        context=name,
        is_active=active,
    )


@pytest.mark.parametrize("dependency", ["fork", "related", "inactive-related", "chain", "self"])
def test_repository_refuses_deleting_referenced_prompt(tmp_path: Path, dependency: str) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    child = _prompt("Child", active=dependency != "inactive-related")
    repo.add(parent)
    if dependency != "self":
        repo.add(child)
    if dependency == "fork":
        repo.record_prompt_fork(parent.id, child.id)
    elif dependency in {"related", "inactive-related"}:
        child.related_prompts = [str(parent.id)]
        repo.update(child)
    elif dependency == "self":
        parent.related_prompts = [str(parent.id)]
        repo.update(parent)
    else:
        chain_id = uuid.uuid4()
        chain = PromptChain(
            id=chain_id,
            name="Synthetic chain",
            description="Test",
            steps=[
                PromptChainStep(
                    id=uuid.uuid4(),
                    chain_id=chain_id,
                    prompt_id=parent.id,
                    order_index=1,
                    input_template="",
                    output_variable="",
                )
            ],
        )
        repo.add_chain(chain)
    with pytest.raises(PromptDeleteBlockedError) as err:
        repo.delete(parent.id)
    expected_kind = "related" if dependency in {"inactive-related", "self"} else dependency
    assert expected_kind in err.value.kinds
    assert repo.get(parent.id).id == parent.id
    if dependency == "fork":
        assert repo.get_prompt_parent_fork(child.id) is not None
    if dependency in {"related", "inactive-related", "self"}:
        assert not any(
            issue.code == "CAT004" for issue in run_catalog_check(repo.list(), []).issues
        )


def test_inactive_parent_cannot_be_forked_even_with_stale_cached_copy(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    repo.add(parent)
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "catalog.db"),
        repository=repo,
    )
    try:
        assert manager.get_prompt(parent.id).is_active
        repo.set_prompt_active(parent.id, active=False, expect_active=True)
        with pytest.raises(PromptVersionError, match="inactive"):
            manager.fork_prompt(parent.id, name="Child")
        assert [prompt.id for prompt in repo.list()] == [parent.id]
        assert repo.list_prompt_children(parent.id) == []
    finally:
        manager.close()


def test_repository_can_delete_unreferenced_prompt_and_reports_missing(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    prompt = _prompt("Independent")
    repo.add(prompt)
    repo.delete(prompt.id)
    with pytest.raises(RepositoryNotFoundError):
        repo.get(prompt.id)


def test_manager_does_not_delete_index_for_referenced_prompt(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    child = _prompt("Child")
    child.related_prompts = [str(parent.id)]
    repo.add(parent)
    repo.add(child)
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "catalog.db"),
        repository=repo,
    )
    try:
        collection = cast("Any", manager.collection)
        collection.add(ids=[str(parent.id)], embeddings=[[0.1, 0.2]])
        with pytest.raises(PromptDeletionBlockedError, match="dependen"):
            manager.delete_prompt(parent.id)
        assert collection.get(ids=[str(parent.id)])["ids"] == [str(parent.id)]
        assert repo.get(parent.id).id == parent.id
    finally:
        manager.close()


def test_manager_reports_partial_if_catalog_write_fails_after_index_removal(
    tmp_path: Path,
) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    repo.add(parent)
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "catalog.db"),
        repository=repo,
    )
    try:
        collection = cast("Any", manager.collection)
        collection.add(ids=[str(parent.id)], embeddings=[[0.1, 0.2]])
        with sqlite3.connect(tmp_path / "catalog.db") as conn:
            conn.execute(
                "CREATE TRIGGER fail_delete BEFORE DELETE ON prompts "
                "BEGIN SELECT RAISE(ABORT, 'private-details'); END"
            )
        with pytest.raises(PromptDeletionPartialError, match="index may have changed") as err:
            manager.delete_prompt(parent.id)
        assert "private-details" not in str(err.value)
        assert repo.get(parent.id).id == parent.id
        assert collection.get(ids=[str(parent.id)])["ids"] == []
    finally:
        manager.close()


def test_manager_reports_partial_if_chroma_deletes_then_raises(tmp_path: Path) -> None:
    class PostDeleteChromaError(ChromaError):
        @classmethod
        @override
        def name(cls) -> str:
            return "post_delete_test"

    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    repo.add(parent)
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "catalog.db"),
        repository=repo,
    )
    try:
        collection = cast("Any", manager.collection)
        collection.add(ids=[str(parent.id)], embeddings=[[0.1, 0.2]])
        original_delete = collection.delete

        def delete_then_raise(*, ids: list[str]) -> None:
            original_delete(ids=ids)
            raise PostDeleteChromaError("SECRET_AFTER_MUTATION")

        collection.delete = delete_then_raise
        with pytest.raises(PromptDeletionPartialError, match="index may have changed") as err:
            manager.delete_prompt(parent.id)
        assert "SECRET_AFTER_MUTATION" not in str(err.value)
        assert repo.get(parent.id).id == parent.id
        assert collection.get(ids=[str(parent.id)])["ids"] == []
    finally:
        manager.close()


def test_manager_holds_sqlite_write_lock_while_index_is_removed(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    child = _prompt("Child")
    repo.add(parent)
    repo.add(child)
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "catalog.db"),
        repository=repo,
    )
    collection = cast("Any", manager.collection)
    collection.add(ids=[str(parent.id)], embeddings=[[0.1, 0.2]])
    attempted = threading.Event()
    finished = threading.Event()
    failures: list[str] = []
    original_delete = collection.delete

    def concurrent_related_writer() -> None:
        try:
            with sqlite3.connect(tmp_path / "catalog.db", timeout=5) as conn:
                attempted.set()
                conn.execute(
                    "UPDATE prompts SET related_prompts = ? WHERE id = ?",
                    ('["' + str(parent.id) + '"]', str(child.id)),
                )
                conn.commit()
        except sqlite3.Error as exc:
            failures.append(type(exc).__name__)
        finally:
            finished.set()

    def delete_with_competitor(*, ids: list[str]) -> None:
        worker = threading.Thread(target=concurrent_related_writer, daemon=True)
        worker.start()
        assert attempted.wait(timeout=2)
        assert not finished.wait(timeout=0.2), "Writer committed during protected delete"
        original_delete(ids=ids)

    collection.delete = delete_with_competitor
    try:
        manager.delete_prompt(parent.id)
        assert finished.wait(timeout=5)
        assert failures == []
        with pytest.raises(RepositoryNotFoundError):
            repo.get(parent.id)
    finally:
        manager.close()


def test_dependency_scan_checks_all_rows_before_returning(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    child = _prompt("Child")
    corrupt = _prompt("Corrupt")
    child.related_prompts = [str(parent.id)]
    for prompt in (parent, child, corrupt):
        repo.add(prompt)
    with sqlite3.connect(tmp_path / "catalog.db") as conn:
        conn.execute(
            "UPDATE prompts SET related_prompts = ? WHERE id = ?",
            ("not-json", str(corrupt.id)),
        )
    with pytest.raises(RepositoryError, match="metadata is invalid"):
        repo.get_prompt_delete_dependencies(parent.id)
    with pytest.raises(RepositoryError, match="metadata is invalid"):
        repo.delete(parent.id)


def test_legacy_non_uuid_relation_does_not_block_unrelated_delete(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Independent")
    legacy = _prompt("Legacy")
    legacy.related_prompts = ["historical-title"]
    repo.add(parent)
    repo.add(legacy)
    assert repo.get_prompt_delete_dependencies(parent.id) == ()
    repo.delete(parent.id)
    with pytest.raises(RepositoryNotFoundError):
        repo.get(parent.id)


def test_fork_rejects_corrupt_catalog_before_creating_unlinked_child(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "catalog.db"))
    parent = _prompt("Parent")
    corrupt = _prompt("Corrupt")
    repo.add(parent)
    repo.add(corrupt)
    with sqlite3.connect(tmp_path / "catalog.db") as conn:
        conn.execute(
            "UPDATE prompts SET related_prompts = ? WHERE id = ?",
            ("SECRET_NOT_JSON", str(corrupt.id)),
        )
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "catalog.db"),
        repository=repo,
    )
    try:
        with pytest.raises(PromptVersionError, match="catalog relations") as err:
            manager.fork_prompt(parent.id, name="Never created")
        assert "SECRET_NOT_JSON" not in str(err.value)
        with sqlite3.connect(tmp_path / "catalog.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 2
    finally:
        manager.close()
