import pytest
import tempfile

from gateway.rag.chunker import TextChunker
from gateway.rag.models import Document
from gateway.rag.service import RAGService
from gateway.rag.sqlite_vector_store import SQLiteVectorStore
from gateway.rag.retrievers import RetrievalStrategy, MetadataFilter


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


def test_range_filtering_on_priority(rag_service):
    """Test filtering documents by priority range."""
    docs = [
        Document(
            document_id="doc-1",
            text="Critical bug needs immediate attention",
            metadata={"priority": 1, "team": "backend"}
        ),
        Document(
            document_id="doc-2",
            text="Medium priority feature request",
            metadata={"priority": 3, "team": "frontend"}
        ),
        Document(
            document_id="doc-3",
            text="Low priority documentation update",
            metadata={"priority": 5, "team": "docs"}
        ),
    ]

    for doc in docs:
        rag_service.ingest(doc)

    # Find high priority items (1-2)
    results, _ = rag_service.search(
        query="bug feature documentation",
        top_k=10,
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=MetadataFilter(values={
            "priority": {"$lte": 2}
        }),
    )

    assert len(results) == 1
    assert results[0].chunk.metadata["priority"] == 1


def test_in_operator_for_teams(rag_service):
    """Test $in operator for filtering by multiple teams."""
    docs = [
        Document(
            document_id="doc-1",
            text="AI team working on embeddings",
            metadata={"team": "ai"}
        ),
        Document(
            document_id="doc-2",
            text="ML team training models",
            metadata={"team": "ml"}
        ),
        Document(
            document_id="doc-3",
            text="Platform team managing infrastructure",
            metadata={"team": "platform"}
        ),
    ]

    for doc in docs:
        rag_service.ingest(doc)

    # Find AI or ML team documents
    results, _ = rag_service.search(
        query="team working",
        top_k=10,
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=MetadataFilter(values={
            "team": {"$in": ["ai", "ml"]}
        }),
    )

    assert len(results) == 2
    teams = {r.chunk.metadata["team"] for r in results}
    assert teams == {"ai", "ml"}


def test_nin_operator_excludes_archived(rag_service):
    """Test $nin operator to exclude archived documents."""
    docs = [
        Document(
            document_id="doc-1",
            text="Active project documentation",
            metadata={"status": "active"}
        ),
        Document(
            document_id="doc-2",
            text="Pending review documentation",
            metadata={"status": "pending"}
        ),
        Document(
            document_id="doc-3",
            text="Archived old documentation",
            metadata={"status": "archived"}
        ),
    ]

    for doc in docs:
        rag_service.ingest(doc)

    # Exclude archived
    results, _ = rag_service.search(
        query="documentation",
        top_k=10,
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=MetadataFilter(values={
            "status": {"$nin": ["archived", "deleted"]}
        }),
    )

    assert len(results) == 2
    statuses = {r.chunk.metadata["status"] for r in results}
    assert "archived" not in statuses


def test_combined_filters(rag_service):
    """Test combining multiple filter conditions."""
    docs = [
        Document(
            document_id="doc-1",
            text="High priority AI project",
            metadata={"priority": 1, "team": "ai", "status": "active"}
        ),
        Document(
            document_id="doc-2",
            text="Low priority AI project",
            metadata={"priority": 5, "team": "ai", "status": "active"}
        ),
        Document(
            document_id="doc-3",
            text="High priority platform project",
            metadata={"priority": 1, "team": "platform", "status": "active"}
        ),
        Document(
            document_id="doc-4",
            text="High priority archived AI project",
            metadata={"priority": 1, "team": "ai", "status": "archived"}
        ),
    ]

    for doc in docs:
        rag_service.ingest(doc)

    # Find: team=ai AND priority<=2 AND status != archived
    results, _ = rag_service.search(
        query="project",
        top_k=10,
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=MetadataFilter(values={
            "team": "ai",
            "priority": {"$lte": 2},
            "status": {"$ne": "archived"},
        }),
    )

    assert len(results) == 1
    assert results[0].chunk.metadata["team"] == "ai"
    assert results[0].chunk.metadata["priority"] == 1
    assert results[0].chunk.metadata["status"] == "active"


def test_date_range_filtering(rag_service):
    """Test filtering by date ranges."""
    docs = [
        Document(
            document_id="doc-1",
            text="January 2024 report",
            metadata={"created_at": "2024-01-15"}
        ),
        Document(
            document_id="doc-2",
            text="June 2024 report",
            metadata={"created_at": "2024-06-15"}
        ),
        Document(
            document_id="doc-3",
            text="December 2024 report",
            metadata={"created_at": "2024-12-15"}
        ),
    ]

    for doc in docs:
        rag_service.ingest(doc)

    # Find Q2-Q3 2024 reports (Apr-Sep)
    results, _ = rag_service.search(
        query="report",
        top_k=10,
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=MetadataFilter(values={
            "created_at": {
                "$gte": "2024-04-01",
                "$lt": "2024-10-01",
            }
        }),
    )

    assert len(results) == 1
    assert results[0].chunk.metadata["created_at"] == "2024-06-15"


def test_no_filter_returns_all(rag_service):
    """Test that no filter returns all results."""
    docs = [
        Document(
            document_id=f"doc-{i}",
            text=f"Document {i}",
            metadata={"index": i}
        )
        for i in range(3)
    ]

    for doc in docs:
        rag_service.ingest(doc)

    results, _ = rag_service.search(
        query="document",
        top_k=10,
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=None,
    )

    assert len(results) == 3
