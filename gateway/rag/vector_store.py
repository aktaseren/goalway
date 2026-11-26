from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from gateway.rag.models import DocumentChunk, SearchResult

@dataclass
class StoredVector:
    chunk: DocumentChunk
    embedding: list[float]


class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._vectors: list[StoredVector] = []

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings"
            )

        for chunk, embedding in zip(chunks, embeddings):
            self._vectors.append(
                StoredVector(
                    chunk=chunk,
                    embedding=embedding,
                )
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        scored_results = [
            SearchResult(
                chunk=item.chunk,
                score=self._cosine_similarity(
                    query_embedding,
                    item.embedding,
                ),
            )
            for item in self._vectors
        ]

        scored_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return scored_results[:top_k]

    @staticmethod
    def _cosine_similarity(
        first: list[float],
        second: list[float],
    ) -> float:
        if len(first) != len(second):
            raise ValueError("Embedding dimensions must match")

        dot_product = sum(
            left * right
            for left, right in zip(first, second)
        )

        first_norm = math.sqrt(
            sum(value * value for value in first)
        )
        second_norm = math.sqrt(
            sum(value * value for value in second)
        )

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return dot_product / (first_norm * second_norm)