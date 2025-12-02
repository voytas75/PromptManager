"""Response style helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-02 - Extract response style APIs into mixin for modularisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import (
    ResponseStyleNotFoundError,
    ResponseStyleStorageError,
)
from ..repository import RepositoryError, RepositoryNotFoundError

if TYPE_CHECKING:
    from uuid import UUID

    from models.response_style import ResponseStyle

    from ..repository import PromptRepository

__all__ = ["ResponseStyleSupport"]


class ResponseStyleSupport:
    """Mixin exposing response style CRUD helpers backed by the repository."""

    _repository: PromptRepository

    def list_response_styles(
        self,
        *,
        include_inactive: bool = False,
        search: str | None = None,
    ) -> list[ResponseStyle]:
        """Return stored response styles ordered by name."""
        try:
            return self._repository.list_response_styles(
                include_inactive=include_inactive,
                search=search,
            )
        except RepositoryError as exc:
            raise ResponseStyleStorageError("Unable to list response styles") from exc

    def get_response_style(self, style_id: UUID) -> ResponseStyle:
        """Return a single response style by identifier."""
        try:
            return self._repository.get_response_style(style_id)
        except RepositoryNotFoundError as exc:
            raise ResponseStyleNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise ResponseStyleStorageError(f"Unable to load response style {style_id}") from exc

    def create_response_style(self, style: ResponseStyle) -> ResponseStyle:
        """Persist a new response style record."""
        style.touch()
        try:
            return self._repository.add_response_style(style)
        except RepositoryError as exc:
            raise ResponseStyleStorageError(f"Failed to persist response style {style.id}") from exc

    def update_response_style(self, style: ResponseStyle) -> ResponseStyle:
        """Update an existing response style record."""
        style.touch()
        try:
            return self._repository.update_response_style(style)
        except RepositoryNotFoundError as exc:
            raise ResponseStyleNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise ResponseStyleStorageError(f"Failed to update response style {style.id}") from exc

    def delete_response_style(self, style_id: UUID) -> None:
        """Delete a stored response style."""
        try:
            self._repository.delete_response_style(style_id)
        except RepositoryNotFoundError as exc:
            raise ResponseStyleNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise ResponseStyleStorageError(f"Failed to delete response style {style_id}") from exc
