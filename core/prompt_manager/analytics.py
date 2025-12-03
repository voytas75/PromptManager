"""Category metrics and embedding diagnostics mixin for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract category health and embedding diagnostics helpers to mixin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..embedding import EmbeddingGenerationError
from ..exceptions import PromptManagerError, PromptStorageError
from ..repository import RepositoryError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from models.prompt_model import Prompt

    from ..category_registry import CategoryRegistry
    from ..embedding import EmbeddingProvider
    from ..repository import PromptRepository
    from . import PromptManager as _PromptManager
else:
    _PromptManager = Any

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyticsMixin",
    "CategoryHealth",
    "EmbeddingDiagnostics",
    "EmbeddingDimensionMismatch",
    "MissingEmbedding",
]


def _parse_timestamp(value: Any) -> datetime | None:
    """Return a timezone-aware datetime when parsing succeeds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@dataclass(slots=True)
class EmbeddingDimensionMismatch:
    """Stored prompt embedding vector that no longer matches the reference dimension."""

    prompt_id: UUID
    prompt_name: str
    stored_dimension: int


@dataclass(slots=True)
class MissingEmbedding:
    """Prompt record that is missing a persisted embedding vector."""

    prompt_id: UUID
    prompt_name: str


@dataclass(slots=True)
class EmbeddingDiagnostics:
    """Summary of embedding backend health and stored vector consistency."""

    backend_ok: bool
    backend_message: str
    backend_dimension: int | None
    inferred_dimension: int | None
    chroma_ok: bool
    chroma_message: str
    chroma_count: int | None
    repository_total: int
    prompts_with_embeddings: int
    missing_prompts: list[MissingEmbedding]
    mismatched_prompts: list[EmbeddingDimensionMismatch]
    consistent_counts: bool | None


@dataclass(slots=True)
class CategoryHealth:
    """Aggregated prompt and execution metrics for a category."""

    slug: str
    label: str
    total_prompts: int
    active_prompts: int
    success_rate: float | None
    last_executed_at: datetime | None


class AnalyticsMixin:
    """Analytics helpers shared between Prompt Manager components."""

    _repository: PromptRepository
    _category_registry: CategoryRegistry
    _embedding_provider: EmbeddingProvider

    def get_category_health(self: _PromptManager) -> list[CategoryHealth]:
        """Return prompt and execution health metrics for each category."""
        try:
            prompt_counts = self._repository.get_category_prompt_counts()
            execution_stats = self._repository.get_category_execution_statistics()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to compute category health metrics") from exc

        categories = {
            category.slug: category
            for category in self._category_registry.all(include_archived=True)
        }
        slug_keys = set(prompt_counts.keys()) | set(execution_stats.keys()) | set(categories.keys())
        if not slug_keys:
            slug_keys.add("")

        results: list[CategoryHealth] = []
        for slug in sorted(slug_keys or {""}):
            category = categories.get(slug)
            label = (
                category.label
                if category
                else (slug.replace("-", " ").title() if slug else "Uncategorised")
            )
            counts = prompt_counts.get(slug, {"total_prompts": 0, "active_prompts": 0})
            stats = execution_stats.get(slug)
            success_rate: float | None = None
            last_executed_at: datetime | None = None
            if stats:
                total_runs = int(stats.get("total_runs", 0) or 0)
                success_runs = int(stats.get("success_runs", 0) or 0)
                success_rate = success_runs / total_runs if total_runs else None
                last_executed_at = _parse_timestamp(stats.get("last_executed_at"))
            results.append(
                CategoryHealth(
                    slug=slug or "",
                    label=label,
                    total_prompts=int(counts.get("total_prompts", 0) or 0),
                    active_prompts=int(counts.get("active_prompts", 0) or 0),
                    success_rate=success_rate,
                    last_executed_at=last_executed_at,
                )
            )

        return results

    def diagnose_embeddings(
        self,
        *,
        sample_text: str = "Prompt Manager diagnostics probe",
    ) -> EmbeddingDiagnostics:
        """Return embedding backend health and stored vector consistency details."""
        provider = getattr(self, "_embedding_provider", None)
        if provider is None:
            raise PromptManagerError("Embedding provider is not configured.")

        backend_ok = True
        backend_message = "Embedding backend reachable."
        backend_dimension: int | None = None
        try:
            probe_vector = provider.embed(sample_text)
            backend_dimension = len(probe_vector)
            if backend_dimension == 0:
                backend_ok = False
                backend_message = "Embedding backend returned an empty vector."
        except EmbeddingGenerationError as exc:
            backend_ok = False
            backend_message = f"Unable to generate embeddings: {exc}"
        except Exception as exc:  # noqa: BLE001 - defensive diagnostics surface
            backend_ok = False
            backend_message = f"Unexpected embedding backend error: {exc}"

        chroma_ok = False
        chroma_message = "Chroma collection unavailable."
        chroma_count: int | None = None
        try:
            collection = self.collection
        except PromptManagerError as exc:
            chroma_message = str(exc)
        else:
            try:
                chroma_count = int(collection.count())
                chroma_ok = True
                chroma_message = "Chroma collection reachable."
            except Exception as exc:  # noqa: BLE001 - defensive diagnostics surface
                chroma_message = f"Unable to query Chroma collection: {exc}"

        try:
            prompts: Sequence[Prompt] = self._repository.list()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to load prompts for embedding diagnostics") from exc

        missing_prompts: list[MissingEmbedding] = []
        mismatched: list[EmbeddingDimensionMismatch] = []
        prompts_with_embeddings = 0
        inferred_dimension: int | None = None
        reference_dimension = backend_dimension if backend_dimension else None

        for prompt in prompts:
            vector = prompt.ext4
            if not vector:
                missing_prompts.append(
                    MissingEmbedding(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name or "Unnamed prompt",
                    )
                )
                continue
            vector_values = list(vector)
            stored_dimension = len(vector_values)
            if stored_dimension == 0:
                missing_prompts.append(
                    MissingEmbedding(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name or "Unnamed prompt",
                    )
                )
                continue
            prompts_with_embeddings += 1
            if reference_dimension is None:
                reference_dimension = stored_dimension
                inferred_dimension = stored_dimension
            if reference_dimension is not None and stored_dimension != reference_dimension:
                mismatched.append(
                    EmbeddingDimensionMismatch(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name or "Unnamed prompt",
                        stored_dimension=stored_dimension,
                    )
                )

        consistent_counts: bool | None = None
        if chroma_count is not None:
            consistent_counts = chroma_count == prompts_with_embeddings

        return EmbeddingDiagnostics(
            backend_ok=backend_ok,
            backend_message=backend_message,
            backend_dimension=backend_dimension if backend_ok else None,
            inferred_dimension=inferred_dimension,
            chroma_ok=chroma_ok,
            chroma_message=chroma_message,
            chroma_count=chroma_count,
            repository_total=len(prompts),
            prompts_with_embeddings=prompts_with_embeddings,
            missing_prompts=missing_prompts,
            mismatched_prompts=mismatched,
            consistent_counts=consistent_counts,
        )
