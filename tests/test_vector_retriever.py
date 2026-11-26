import pytest

from gateway.rag.models import (
    DocumentChunk,
    SearchResult,
)
from gateway.rag.retrievers import (
    MetadataFilter,
    VectorRetriever,
)
from gateway.rag.vector_store import VectorStore


class FakeEmbeddingProvider:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        assert texts == ["openshift"]

        return [
            [0.1, 0.2, 0.3],
        ]


class FakeVectorStore(VectorStore):
    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        pass

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        assert query_embedding == [
            0.1,
            0.2,
            0.3,
        ]

        results = [
            SearchResult(
                chunk=DocumentChunk(
                    chunk_id="1",
                    document_id="doc",
                    text="OpenShift AI",
                    position=0,
                    metadata={
                        "team": "ai",
                    },
                ),
                score=0.95,
            ),
            SearchResult(
                chunk=DocumentChunk(
                    chunk_id="2",
                    document_id="doc",
                    text="Kubernetes platform",
                    position=1,
                    metadata={
                        "team": "platform",
                    },
                ),
                score=0.70,
            ),
        ]

        return results[:top_k]

def test_vector_retriever_returns_results():
    retriever = VectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.search(
        query="openshift",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].chunk.chunk_id == "1"
    assert results[0].score == 0.95
    assert results[0].rank == 1
    assert results[0].retriever == "vector"

    assert results[1].chunk.chunk_id == "2"
    assert results[1].rank == 2
def test_vector_retriever_applies_metadata_filter():
    retriever = VectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.search(
        query="openshift",
        top_k=2,
        metadata_filter=MetadataFilter(
            values={
                "team": "platform",
            }
        ),
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "2"
    assert results[0].rank == 1
def test_vector_retriever_requests_extra_candidates_for_filtering():
    class RecordingVectorStore(FakeVectorStore):
        requested_top_k: int | None = None

        def search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
        ) -> list[SearchResult]:
            self.requested_top_k = top_k

            return super().search(
                query_embedding=query_embedding,
                top_k=top_k,
            )

    store = RecordingVectorStore()

    retriever = VectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    retriever.search(
        query="openshift",
        top_k=2,
    )

    assert store.requested_top_k == 8
def test_vector_retriever_rejects_invalid_top_k():
    retriever = VectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.search(
            query="openshift",
            top_k=0,
        )
def test_vector_retriever_requires_one_query_embedding():
    class InvalidEmbeddingProvider:
        def embed(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [
                [0.1],
                [0.2],
            ]

    retriever = VectorRetriever(
        embedding_provider=InvalidEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="must return one query vector",
    ):
        retriever.search(
            query="openshift",
        )