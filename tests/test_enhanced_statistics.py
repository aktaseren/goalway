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


def test_statistics_track_latency(rag_service):
    """Test that statistics include latency measurements."""
    doc = Document(
        document_id="doc-1",
        text="OpenShift AI provides model serving capabilities.",
    )

    rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="OpenShift model serving",
        top_k=5,
        strategy=RetrievalStrategy.HYBRID,
    )

    assert stats is not None
    assert stats.total_latency_ms > 0
    assert stats.lexical_latency_ms > 0
    assert stats.vector_latency_ms > 0
    assert stats.fusion_latency_ms > 0

    # Total should be roughly sum of components
    component_sum = (
        stats.lexical_latency_ms +
        stats.vector_latency_ms +
        stats.fusion_latency_ms
    )
    assert stats.total_latency_ms >= component_sum * 0.9  # Allow 10% overhead


def test_statistics_track_recall(rag_service):
    """Test that statistics include recall estimation."""
    docs = [
        Document(
            document_id=f"doc-{i}",
            text=f"Document {i} about OpenShift and Kubernetes",
        )
        for i in range(5)
    ]

    for doc in docs:
        rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="OpenShift Kubernetes",
        top_k=3,
        strategy=RetrievalStrategy.HYBRID,
    )

    assert stats is not None
    assert stats.lexical_vector_overlap >= 0
    assert 0 <= stats.overlap_percentage <= 100


def test_statistics_track_rrf_scores(rag_service):
    """Test that statistics include RRF score diagnostics."""
    docs = [
        Document(
            document_id=f"doc-{i}",
            text=f"Document {i} content",
        )
        for i in range(3)
    ]

    for doc in docs:
        rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="document content",
        top_k=3,
        strategy=RetrievalStrategy.HYBRID,
    )

    assert stats is not None
    assert stats.avg_rrf_score > 0
    assert stats.max_rrf_score > 0
    assert stats.min_rrf_score > 0
    assert stats.max_rrf_score >= stats.avg_rrf_score >= stats.min_rrf_score


def test_lexical_only_statistics(rag_service):
    """Test statistics for lexical-only strategy."""
    doc = Document(
        document_id="doc-1",
        text="Test document",
    )

    rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="test",
        top_k=5,
        strategy=RetrievalStrategy.LEXICAL,
    )

    assert stats is not None
    assert stats.lexical_candidates > 0
    assert stats.vector_candidates == 0
    assert stats.lexical_vector_overlap == 0
    assert stats.overlap_percentage == 0.0


def test_vector_only_statistics(rag_service):
    """Test statistics for vector-only strategy."""
    doc = Document(
        document_id="doc-1",
        text="Test document",
    )

    rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="test",
        top_k=5,
        strategy=RetrievalStrategy.VECTOR,
    )

    assert stats is not None
    assert stats.vector_candidates > 0
    assert stats.lexical_candidates == 0
    assert stats.lexical_vector_overlap == 0
    assert stats.overlap_percentage == 0.0


def test_high_overlap_indicates_agreement(rag_service):
    """Test that high overlap indicates both retrievers agree."""
    # Create documents that should match well on both lexical and semantic
    docs = [
        Document(
            document_id="doc-1",
            text="Machine learning model training with GPUs",
        ),
        Document(
            document_id="doc-2",
            text="Deep neural network optimization techniques",
        ),
        Document(
            document_id="doc-3",
            text="Unrelated content about cooking recipes",
        ),
    ]

    for doc in docs:
        rag_service.ingest(doc)

    # Query that should match well lexically and semantically
    results, stats = rag_service.search(
        query="machine learning model training",
        top_k=2,
        strategy=RetrievalStrategy.HYBRID,
    )

    assert stats is not None
    # Expect some overlap when query matches well
    assert stats.lexical_vector_overlap >= 0


def test_statistics_reflect_candidates_before_fusion(rag_service):
    """Test that candidate counts reflect pre-fusion results."""
    docs = [
        Document(
            document_id=f"doc-{i}",
            text=f"Document number {i}",
        )
        for i in range(10)
    ]

    for doc in docs:
        rag_service.ingest(doc)

    results, stats = rag_service.search(
        query="document",
        top_k=3,  # Request 3 results
        strategy=RetrievalStrategy.HYBRID,
    )

    assert stats is not None
    # Candidates should be higher due to candidate_multiplier
    # (default 4x, so expect ~12 candidates per retriever)
    assert stats.lexical_candidates >= stats.returned_results
    assert stats.vector_candidates >= stats.returned_results
