"""Semantic search and recommendation helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract search, suggestion, and personalisation mixin.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chromadb.errors import ChromaError

from models.prompt_model import Prompt

from ..embedding import EmbeddingGenerationError
from ..exceptions import PromptManagerError, PromptStorageError
from ..intent_classifier import IntentPrediction, rank_by_hints
from ..repository import RepositoryError, RepositoryNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from models.prompt_model import UserProfile

    from ..intent_classifier import IntentClassifier
    from ..repository import PromptRepository

logger = logging.getLogger(__name__)

__all__ = ["IntentSuggestions", "PromptSearchMixin"]


@dataclass(slots=True)
class IntentSuggestions:
    """Intent-aware search recommendations returned to callers."""

    prediction: IntentPrediction
    prompts: list[Prompt]
    fallback_used: bool = False


class PromptSearchMixin:
    """Shared semantic search and intent suggestion helpers."""

    _embedding_provider: Any  # EmbeddingProvider protocol
    _repository: PromptRepository
    _intent_classifier: IntentClassifier | None
    _user_profile: UserProfile | None

    def search_prompts(
        self,
        query_text: str,
        limit: int = 5,
        where: dict[str, Any] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> list[Prompt]:
        """Search prompts semantically using a text query or embedding."""
        if not query_text and embedding is None:
            raise ValueError("query_text or embedding must be provided")

        collection = self.collection

        query_embedding: list[float]
        if embedding is not None:
            query_embedding = [float(value) for value in embedding]
        else:
            try:
                query_embedding = self._embedding_provider.embed(query_text)
            except EmbeddingGenerationError as exc:
                raise PromptStorageError("Failed to generate query embedding") from exc

        try:
            try:
                results = self._query_chroma(
                    collection,
                    query_embedding,
                    limit=limit,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            except TypeError:
                results = self._query_chroma(
                    collection,
                    query_embedding,
                    limit=limit,
                    where=where,
                    include=None,
                )
        except ChromaError as exc:
            raise PromptStorageError("Failed to query prompts") from exc

        prompts: list[Prompt] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distance_values = self._extract_distances(results, len(ids))

        for prompt_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distance_values,
            strict=False,
        ):
            try:
                prompt_uuid = uuid.UUID(prompt_id)
            except ValueError:
                logger.warning(
                    "Invalid prompt UUID in Chroma results",
                    extra={"prompt_id": prompt_id},
                )
                continue
            try:
                prompt_record = self._repository.get(prompt_uuid)
            except RepositoryNotFoundError:
                record = {"id": prompt_id, "document": document, "metadata": metadata}
                prompt_record = self._hydrate_prompt(record)
            except RepositoryError as exc:
                raise PromptStorageError(
                    f"Failed to hydrate prompt {prompt_id} from SQLite"
                ) from exc

            try:
                if distance is not None:
                    prompt_record.similarity = 1.0 - float(distance)
            except Exception:  # pragma: no cover - defensive
                pass

            prompts.append(prompt_record)

        if any(p.similarity is not None for p in prompts):
            prompts.sort(key=lambda p: p.similarity or 0.0, reverse=True)
        return prompts

    def suggest_prompts(
        self,
        query_text: str,
        *,
        limit: int = 5,
    ) -> IntentSuggestions:
        """Return intent-ranked prompt recommendations for the supplied query."""
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        stripped = query_text.strip()
        if not stripped:
            try:
                baseline = self.repository.list(limit=limit)
            except RepositoryError as exc:
                raise PromptStorageError("Unable to load prompts for suggestions") from exc
            personalised = self._personalize_ranked_prompts(baseline)
            return IntentSuggestions(
                IntentPrediction.general(),
                personalised[:limit],
                fallback_used=True,
            )

        prediction = (
            self._intent_classifier.classify(stripped)
            if self._intent_classifier is not None
            else IntentPrediction.general()
        )

        augmented_query_parts = [stripped]
        if prediction.category_hints:
            augmented_query_parts.append(
                "Intent categories: " + ", ".join(prediction.category_hints)
            )
        if prediction.tag_hints:
            augmented_query_parts.append("Intent tags: " + ", ".join(prediction.tag_hints))
        augmented_query = "\n".join(augmented_query_parts)

        suggestions: list[Prompt] = []
        fallback_used = False
        try:
            raw_results = self.search_prompts(augmented_query, limit=max(limit * 2, 10))
        except PromptManagerError:
            raw_results = []

        ranked = rank_by_hints(
            raw_results,
            category_hints=prediction.category_hints,
            tag_hints=prediction.tag_hints,
        )
        ranked = self._personalize_ranked_prompts(ranked)

        seen_ids: set[uuid.UUID] = set()
        for prompt in ranked:
            if prompt.id in seen_ids:
                continue
            suggestions.append(prompt)
            seen_ids.add(prompt.id)
            if len(suggestions) >= limit:
                break

        if len(suggestions) < limit:
            fallback_used = True
            try:
                fallback_results = self.search_prompts(stripped, limit=max(limit * 2, 10))
            except PromptManagerError:
                fallback_results = []
            else:
                fallback_results = self._personalize_ranked_prompts(fallback_results)
            for prompt in fallback_results:
                if prompt.id in seen_ids:
                    continue
                suggestions.append(prompt)
                seen_ids.add(prompt.id)
                if len(suggestions) >= limit:
                    break

        if not suggestions:
            fallback_used = True
            try:
                suggestions = self.repository.list(limit=limit)
            except RepositoryError as exc:
                raise PromptStorageError("Unable to load prompts for suggestions") from exc

        personalised = self._personalize_ranked_prompts(suggestions)
        return IntentSuggestions(
            prediction=prediction,
            prompts=personalised[:limit],
            fallback_used=fallback_used,
        )

    # Helpers ---------------------------------------------------------- #

    def _hydrate_prompt(self, record: Mapping[str, Any]) -> Prompt:
        """Reconstruct a Prompt model from Chroma metadata."""
        return Prompt.from_chroma(record)

    def _query_chroma(
        self,
        collection: Any,
        query_embedding: Sequence[float],
        *,
        limit: int,
        where: dict[str, Any] | None,
        include: list[str] | None,
    ) -> dict[str, Any]:
        """Execute a Chroma query with defensive include handling."""
        kwargs: dict[str, Any] = {
            "query_texts": None,
            "query_embeddings": [list(query_embedding)],
            "n_results": limit,
            "where": where,
        }
        if include is not None:
            kwargs["include"] = include
        return collection.query(**kwargs)

    def _extract_distances(self, results: dict[str, Any], target_length: int) -> list[float | None]:
        """Normalise returned Chroma distance vectors."""
        distances = results.get("distances")
        if isinstance(distances, list) and distances:
            distance_values = distances[0]
        else:
            distance_values = []
        if len(distance_values) < target_length:
            distance_values = [
                *distance_values,
                *([None] * (target_length - len(distance_values))),
            ]
        return distance_values

    def _personalize_ranked_prompts(self, prompts: Sequence[Prompt]) -> list[Prompt]:
        """Bias prompt order using stored user preferences while preserving stability."""
        if not prompts:
            return []
        profile = self._user_profile
        if profile is None:
            return list(prompts)

        favorite_categories = profile.favorite_categories(limit=5)
        favorite_tags = profile.favorite_tags(limit=8)
        if not favorite_categories and not favorite_tags:
            return list(prompts)

        category_weights = {
            name: (len(favorite_categories) - idx) * 2
            for idx, name in enumerate(favorite_categories)
        }
        tag_weights = {name: len(favorite_tags) - idx for idx, name in enumerate(favorite_tags)}

        scored: list[tuple[float, int, Prompt]] = []
        for index, prompt in enumerate(prompts):
            score = 0.0
            category = (prompt.category or "").strip()
            if category in category_weights:
                score += float(category_weights[category])
            for tag in prompt.tags or []:
                weight = tag_weights.get(tag)
                if weight:
                    score += float(weight)
            scored.append((score, index, prompt))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [prompt for _, _, prompt in scored]
