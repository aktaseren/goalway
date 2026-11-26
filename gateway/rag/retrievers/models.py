from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gateway.rag.models import DocumentChunk


class RetrievalStrategy(str, Enum):
    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class MetadataFilter:
    """
    Advanced metadata filtering with operator support.

    Supports:
    - Exact match: {"team": "ai"}
    - Equality: {"team": {"$eq": "ai"}}
    - Not equal: {"team": {"$ne": "platform"}}
    - Less than: {"priority": {"$lt": 5}}
    - Less than or equal: {"priority": {"$lte": 5}}
    - Greater than: {"priority": {"$gt": 3}}
    - Greater than or equal: {"priority": {"$gte": 3}}
    - In list: {"team": {"$in": ["ai", "ml"]}}
    - Not in list: {"status": {"$nin": ["archived", "deleted"]}}
    """

    values: dict[str, Any] = field(default_factory=dict)

    def matches(self, metadata: dict[str, Any]) -> bool:
        """
        Check if metadata matches all filter conditions.

        Args:
            metadata: Document metadata to check

        Returns:
            True if all filter conditions are satisfied
        """
        return all(
            self._matches_condition(key, condition, metadata)
            for key, condition in self.values.items()
        )

    def _matches_condition(
        self,
        key: str,
        condition: Any,
        metadata: dict[str, Any],
    ) -> bool:
        """
        Check if a single filter condition matches.

        Args:
            key: Metadata field name
            condition: Filter condition (value or operator dict)
            metadata: Document metadata

        Returns:
            True if condition is satisfied
        """
        metadata_value = metadata.get(key)

        # Simple equality check (backward compatible)
        if not isinstance(condition, dict):
            return metadata_value == condition

        # Operator-based filtering
        for operator, expected_value in condition.items():
            if operator == "$eq":
                if metadata_value != expected_value:
                    return False

            elif operator == "$ne":
                if metadata_value == expected_value:
                    return False

            elif operator == "$lt":
                if metadata_value is None or metadata_value >= expected_value:
                    return False

            elif operator == "$lte":
                if metadata_value is None or metadata_value > expected_value:
                    return False

            elif operator == "$gt":
                if metadata_value is None or metadata_value <= expected_value:
                    return False

            elif operator == "$gte":
                if metadata_value is None or metadata_value < expected_value:
                    return False

            elif operator == "$in":
                if metadata_value not in expected_value:
                    return False

            elif operator == "$nin":
                if metadata_value in expected_value:
                    return False

            else:
                # Unknown operator - treat as false for safety
                return False

        return True


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int
    retriever: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalStats:
    strategy: RetrievalStrategy
    query: str
    returned_results: int
    lexical_candidates: int = 0
    vector_candidates: int = 0
    filtered_candidates: int = 0

    # Latency tracking (milliseconds)
    lexical_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Recall estimation
    lexical_vector_overlap: int = 0  # How many results appear in both
    overlap_percentage: float = 0.0  # % of results from both sources

    # Ranking diagnostics
    avg_rrf_score: float = 0.0
    max_rrf_score: float = 0.0
    min_rrf_score: float = 0.0

    def with_latency(
        self,
        lexical_ms: float = 0.0,
        vector_ms: float = 0.0,
        fusion_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> "RetrievalStats":
        """Create a new stats instance with latency information."""
        return RetrievalStats(
            strategy=self.strategy,
            query=self.query,
            returned_results=self.returned_results,
            lexical_candidates=self.lexical_candidates,
            vector_candidates=self.vector_candidates,
            filtered_candidates=self.filtered_candidates,
            lexical_latency_ms=lexical_ms,
            vector_latency_ms=vector_ms,
            fusion_latency_ms=fusion_ms,
            total_latency_ms=total_ms,
            lexical_vector_overlap=self.lexical_vector_overlap,
            overlap_percentage=self.overlap_percentage,
            avg_rrf_score=self.avg_rrf_score,
            max_rrf_score=self.max_rrf_score,
            min_rrf_score=self.min_rrf_score,
        )

    def with_recall(
        self,
        overlap: int,
        overlap_pct: float,
    ) -> "RetrievalStats":
        """Create a new stats instance with recall information."""
        return RetrievalStats(
            strategy=self.strategy,
            query=self.query,
            returned_results=self.returned_results,
            lexical_candidates=self.lexical_candidates,
            vector_candidates=self.vector_candidates,
            filtered_candidates=self.filtered_candidates,
            lexical_latency_ms=self.lexical_latency_ms,
            vector_latency_ms=self.vector_latency_ms,
            fusion_latency_ms=self.fusion_latency_ms,
            total_latency_ms=self.total_latency_ms,
            lexical_vector_overlap=overlap,
            overlap_percentage=overlap_pct,
            avg_rrf_score=self.avg_rrf_score,
            max_rrf_score=self.max_rrf_score,
            min_rrf_score=self.min_rrf_score,
        )

    def with_ranking(
        self,
        avg_score: float,
        max_score: float,
        min_score: float,
    ) -> "RetrievalStats":
        """Create a new stats instance with ranking diagnostics."""
        return RetrievalStats(
            strategy=self.strategy,
            query=self.query,
            returned_results=self.returned_results,
            lexical_candidates=self.lexical_candidates,
            vector_candidates=self.vector_candidates,
            filtered_candidates=self.filtered_candidates,
            lexical_latency_ms=self.lexical_latency_ms,
            vector_latency_ms=self.vector_latency_ms,
            fusion_latency_ms=self.fusion_latency_ms,
            total_latency_ms=self.total_latency_ms,
            lexical_vector_overlap=self.lexical_vector_overlap,
            overlap_percentage=self.overlap_percentage,
            avg_rrf_score=avg_score,
            max_rrf_score=max_score,
            min_rrf_score=min_score,
        )