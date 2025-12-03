"""User state bootstrap helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract history tracker and user profile initialisation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..repository import RepositoryError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from models.prompt_model import UserProfile

    from ..history_tracker import HistoryTracker
    from ..repository import PromptRepository

logger = logging.getLogger(__name__)

__all__ = ["UserStateMixin"]


class UserStateMixin:
    """Mixin responsible for establishing history trackers and user profile state."""

    _repository: PromptRepository
    _history_tracker: HistoryTracker | None
    _user_profile: UserProfile | None

    def _initialise_user_state(
        self,
        *,
        history_tracker: HistoryTracker | None,
        user_profile: UserProfile | None,
    ) -> None:
        """Wire runtime trackers and optionally hydrate the persisted profile."""
        self._history_tracker = history_tracker
        if user_profile is not None:
            self._user_profile = user_profile
            return
        try:
            self._user_profile = self._repository.get_user_profile()
        except RepositoryError:
            logger.warning("Unable to load persisted user profile", exc_info=True)
            self._user_profile = None
