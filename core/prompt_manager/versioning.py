"""Prompt versioning and fork helpers for Prompt Manager.

Updates:
  v0.1.1 - 2026-04-10 - Reset forked prompts to a fresh visible version baseline.
  v0.1.0 - 2025-12-03 - Extract versioning, diff, and fork APIs into mixin.
"""

from __future__ import annotations

import difflib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from models.prompt_model import Prompt, PromptForkLink, PromptVersion

from ..exceptions import (
    PromptVersionError,
    PromptVersionNotFoundError,
)
from ..repository import RepositoryError, RepositoryNotFoundError

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from ..repository import PromptRepository

__all__ = ["PromptVersionDiff", "PromptVersionMixin"]


@dataclass(slots=True)
class PromptVersionDiff:
    """Diff payload surfaced when comparing two prompt versions."""

    prompt_id: uuid.UUID
    base_version: PromptVersion
    target_version: PromptVersion
    changed_fields: dict[str, dict[str, Any]]
    body_diff: str


class PromptVersionMixin:
    """Prompt versioning, diff, and fork helpers."""

    _repository: PromptRepository

    # Public APIs ------------------------------------------------------ #

    def list_prompt_versions(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int | None = None,
    ) -> list[PromptVersion]:
        """Return committed versions for the specified prompt."""
        try:
            return self._repository.list_prompt_versions(prompt_id, limit=limit)
        except RepositoryError as exc:
            raise PromptVersionError("Unable to load prompt versions") from exc

    def get_prompt_version(self, version_id: int) -> PromptVersion:
        """Return a stored prompt version by identifier."""
        try:
            return self._repository.get_prompt_version(version_id)
        except RepositoryNotFoundError as exc:
            raise PromptVersionNotFoundError(f"Prompt version {version_id} not found") from exc
        except RepositoryError as exc:
            raise PromptVersionError(f"Unable to load prompt version {version_id}") from exc

    def get_latest_prompt_version(self, prompt_id: uuid.UUID) -> PromptVersion | None:
        """Return the most recent version for the prompt, if one exists."""
        try:
            return self._repository.get_prompt_latest_version(prompt_id)
        except RepositoryError as exc:
            raise PromptVersionError("Unable to load latest prompt version") from exc

    def diff_prompt_versions(
        self,
        base_version_id: int,
        target_version_id: int,
    ) -> PromptVersionDiff:
        """Return a structured diff between two version snapshots."""
        base_version = self.get_prompt_version(base_version_id)
        target_version = self.get_prompt_version(target_version_id)
        if base_version.prompt_id != target_version.prompt_id:
            raise PromptVersionError("Versions belong to different prompts")

        changed_fields: dict[str, dict[str, Any]] = {}
        keys = set(base_version.snapshot.keys()) | set(target_version.snapshot.keys())
        for key in sorted(keys):
            base_value = base_version.snapshot.get(key)
            target_value = target_version.snapshot.get(key)
            if base_value != target_value:
                changed_fields[key] = {"from": base_value, "to": target_value}

        base_body = str(base_version.snapshot.get("context") or "")
        target_body = str(target_version.snapshot.get("context") or "")
        body_diff = self._render_text_diff(
            base_body,
            target_body,
            label_a=f"v{base_version.version_number}",
            label_b=f"v{target_version.version_number}",
        )

        return PromptVersionDiff(
            prompt_id=base_version.prompt_id,
            base_version=base_version,
            target_version=target_version,
            changed_fields=changed_fields,
            body_diff=body_diff,
        )

    def restore_prompt_version(
        self,
        version_id: int,
        *,
        commit_message: str | None = None,
    ) -> Prompt:
        """Replace the live prompt with the contents of the specified version."""
        version = self.get_prompt_version(version_id)
        prompt = version.to_prompt()
        prompt.last_modified = datetime.now(UTC)
        message = commit_message or f"Restore version {version.version_number}"
        return cast("Any", self).update_prompt(prompt, commit_message=message)

    def merge_prompt_versions(
        self,
        prompt_id: uuid.UUID,
        *,
        base_version_id: int,
        incoming_version_id: int,
        persist: bool = False,
        commit_message: str | None = None,
    ) -> tuple[Prompt, list[str]]:
        """Perform a simple three-way merge and optionally persist the result."""
        base_version = self.get_prompt_version(base_version_id)
        incoming_version = self.get_prompt_version(incoming_version_id)
        if base_version.prompt_id != prompt_id or incoming_version.prompt_id != prompt_id:
            raise PromptVersionError("Versions do not belong to the requested prompt")

        current_prompt = cast("Any", self).get_prompt(prompt_id)
        merged_snapshot = dict(current_prompt.to_record())
        base_snapshot = base_version.snapshot
        incoming_snapshot = incoming_version.snapshot
        conflicts: list[str] = []
        merge_fields = {
            "context",
            "description",
            "tags",
            "category",
            "language",
            "example_input",
            "example_output",
            "scenarios",
            "ext1",
            "ext2",
            "ext3",
            "ext4",
            "ext5",
        }

        for field in merge_fields:
            base_value = base_snapshot.get(field)
            incoming_value = incoming_snapshot.get(field)
            current_value = merged_snapshot.get(field)
            if incoming_value == base_value:
                continue
            if current_value == base_value:
                merged_snapshot[field] = incoming_value
            elif current_value == incoming_value:
                continue
            else:
                merged_snapshot[field] = incoming_value
                conflicts.append(field)

        merged_prompt = Prompt.from_record(merged_snapshot)
        merged_prompt.last_modified = datetime.now(UTC)

        if not persist:
            return merged_prompt, conflicts

        message = commit_message or (
            f"Merge versions {base_version.version_number} -> {incoming_version.version_number}"
        )
        stored = cast("Any", self).update_prompt(merged_prompt, commit_message=message)
        return stored, conflicts

    def fork_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        name: str | None = None,
        commit_message: str | None = None,
    ) -> Prompt:
        """Create a new prompt based on the referenced prompt."""
        source_prompt = cast("Any", self).get_prompt(prompt_id)
        now = datetime.now(UTC)
        related_prompts = list(source_prompt.related_prompts)
        source_id_text = str(source_prompt.id)
        if source_id_text not in related_prompts:
            related_prompts.append(source_id_text)

        fork_record = json.loads(json.dumps(source_prompt.to_record(), ensure_ascii=False))
        forked_prompt = Prompt.from_record(fork_record)
        forked_prompt.id = uuid.uuid4()
        forked_prompt.name = name or f"{source_prompt.name} (fork)"
        forked_prompt.version = "1"
        forked_prompt.last_modified = now
        forked_prompt.created_at = now
        forked_prompt.usage_count = 0
        forked_prompt.rating_count = 0
        forked_prompt.rating_sum = 0.0
        forked_prompt.similarity = None
        forked_prompt.source = "fork"
        forked_prompt.related_prompts = related_prompts
        forked_prompt.quality_score = source_prompt.quality_score
        stored: Prompt | None = None
        try:
            stored = cast(
                "Prompt",
                cast("Any", self).create_prompt(
                    forked_prompt,
                    commit_message=commit_message or f"Forked from {source_prompt.name}",
                ),
            )
            lineage = self._repository.record_prompt_fork(source_prompt.id, stored.id)
            if lineage.source_prompt_id != source_prompt.id or lineage.child_prompt_id != stored.id:
                raise RepositoryError("Recorded prompt fork lineage does not match the new prompt")
        except Exception as exc:
            rollback_prompt_id: uuid.UUID | None = stored.id if stored is not None else None
            if rollback_prompt_id is None:
                try:
                    self._repository.get(forked_prompt.id)
                except RepositoryNotFoundError:
                    pass
                except RepositoryError as lookup_exc:
                    raise PromptVersionError(
                        "Failed to determine whether the incomplete prompt fork was persisted"
                    ) from lookup_exc
                else:
                    rollback_prompt_id = forked_prompt.id
            if rollback_prompt_id is not None:
                try:
                    cast("Any", self).delete_prompt(rollback_prompt_id)
                except Exception as rollback_exc:
                    raise PromptVersionError(
                        f"Failed to complete prompt fork and roll back fork {rollback_prompt_id}"
                    ) from rollback_exc
            if isinstance(exc, RepositoryError):
                raise PromptVersionError("Failed to record prompt fork relationship") from exc
            raise

        return stored

    def list_prompt_forks(self, prompt_id: uuid.UUID) -> list[PromptForkLink]:
        """Return lineage entries for children derived from the prompt."""
        try:
            return self._repository.list_prompt_children(prompt_id)
        except RepositoryError as exc:
            raise PromptVersionError("Unable to load prompt forks") from exc

    def get_prompt_parent_fork(self, prompt_id: uuid.UUID) -> PromptForkLink | None:
        """Return the recorded parent for a forked prompt, if any."""
        try:
            return self._repository.get_prompt_parent_fork(prompt_id)
        except RepositoryError as exc:
            raise PromptVersionError("Unable to load fork lineage") from exc

    # Internal helpers ------------------------------------------------- #

    def _commit_prompt_version(
        self,
        prompt: Prompt,
        *,
        commit_message: str | None = None,
        parent_version_id: int | None = None,
    ) -> PromptVersion:
        """Persist a version snapshot for the provided prompt."""
        try:
            return self._repository.record_prompt_version(
                prompt,
                commit_message=commit_message,
                parent_version_id=parent_version_id,
            )
        except RepositoryError as exc:
            raise PromptVersionError(f"Failed to record version for prompt {prompt.id}") from exc

    @staticmethod
    def _render_text_diff(
        before: str,
        after: str,
        *,
        label_a: str = "before",
        label_b: str = "after",
    ) -> str:
        """Return a unified diff for the provided text blocks."""
        diff = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=label_a,
            tofile=label_b,
            lineterm="",
        )
        return "\n".join(diff)
