from gateway.rag.embeddings.base import (
    EmbeddingProvider,
)
from gateway.rag.retrievers.base import Retriever
from gateway.rag.retrievers.models import (
    MetadataFilter,
    RetrievalResult,
)
from gateway.rag.vector_store import VectorStore


class VectorRetriever(Retriever):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    @property
    def name(self) -> str:
        return "vector"

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        embeddings = self.embedding_provider.embed(
            [query]
        )

        if len(embeddings) != 1:
            raise ValueError(
                "Embedding provider must return one query vector"
            )

        candidates = self.vector_store.search(
            query_embedding=embeddings[0],
            top_k=max(top_k * 4, top_k),
        )

        results: list[RetrievalResult] = []

        for candidate in candidates:
            chunk_metadata = getattr(
                candidate.chunk,
                "metadata",
                {},
            ) or {}

            if (
                metadata_filter is not None
                and not metadata_filter.matches(
                    chunk_metadata
                )
            ):
                continue

            results.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    score=candidate.score,
                    rank=len(results) + 1,
                    retriever=self.name,
                )
            )

            if len(results) >= top_k:
                break

        return results