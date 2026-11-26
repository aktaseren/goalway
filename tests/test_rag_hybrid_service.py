import pytest
import tempfile

from gateway.rag.chunker import TextChunker
from gateway.rag.models import Document
from gateway.rag.service import RAGService
from gateway.rag.sqlite_vector_store import SQLiteVectorStore
from gateway.rag.retrievers import RetrievalStrategy


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(i) for i in range(384)]
            for _ in texts
        ]


@pytest.fixture
def rag_service():
    with tempfile.NamedTemporaryFile(delete=False) as db:
        vector_store = SQLiteVectorStore(db.name)
        service = RAGService(
            chunker=TextChunker(chunk_size=100, overlap=20),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=vector_store,
        )
        yield service


def test_hybrid_search_end_to_end(rag_service):
    doc1 = Document(
        document_id="doc-1",
        text="OpenShift AI provides model serving capabilities for machine learning models.",
        metadata={"team": "ai"},
    )

    doc2 = Document(
        document_id="doc-2",
        text="Kubernetes orchestrates containerized applications across clusters.",
        metadata={"team": "platform"},
    )

    rag_service.ingest(doc1)
    rag_service.ingest(doc2)

    results, stats = rag_service.search(
        query="OpenShift model serving",
        top_k=2,
        strategy=RetrievalStrategy.HYBRID,
    )

    assert len(results) > 0
    assert stats is not None
    assert stats.strategy == RetrievalStrategy.HYBRID
    assert stats.returned_results == len(results)
    assert stats.lexical_candidates > 0
    assert stats.vector_candidates > 0


def test_lexical_search_strategy(rag_service):
    doc = Document(
        document_id="doc-1",
        text="Red Hat OpenShift is an enterprise Kubernetes platform.",
    )

    rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="OpenShift Kubernetes",
        top_k=5,
        strategy=RetrievalStrategy.LEXICAL,
    )

    assert len(results) > 0
    assert stats is not None
    assert stats.strategy == RetrievalStrategy.LEXICAL
    assert stats.lexical_candidates > 0
    assert stats.vector_candidates == 0


def test_vector_search_strategy(rag_service):
    doc = Document(
        document_id="doc-1",
        text="Machine learning models need serving infrastructure.",
    )

    rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="AI model deployment",
        top_k=5,
        strategy=RetrievalStrategy.VECTOR,
    )

    assert len(results) > 0
    assert stats is not None
    assert stats.strategy == RetrievalStrategy.VECTOR
    assert stats.vector_candidates > 0
    assert stats.lexical_candidates == 0


def test_metadata_filtering_in_hybrid_search(rag_service):
    from gateway.rag.retrievers import MetadataFilter

    doc1 = Document(
        document_id="doc-1",
        text="AI team documentation about models.",
        metadata={"team": "ai"},
    )

    doc2 = Document(
        document_id="doc-2",
        text="Platform team documentation about infrastructure.",
        metadata={"team": "platform"},
    )

    rag_service.ingest(doc1)
    rag_service.ingest(doc2)

    results, stats = rag_service.search(
        query="documentation",
        top_k=5,
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=MetadataFilter(values={"team": "ai"}),
    )

    assert all(
        result.chunk.metadata.get("team") == "ai"
        for result in results
    )
    assert stats is not None
    assert stats.strategy == RetrievalStrategy.HYBRID


def test_invalid_strategy_raises_error(rag_service):
    doc = Document(
        document_id="doc-1",
        text="Test document",
    )

    rag_service.ingest(doc)

    with pytest.raises(ValueError, match="Unknown strategy"):
        rag_service.search(
            query="test",
            strategy="invalid_strategy",
        )
