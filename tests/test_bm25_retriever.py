from gateway.rag.models import DocumentChunk
from gateway.rag.retrievers import (
    BM25Retriever,
    MetadataFilter,
)


def chunk(
    chunk_id: str,
    text: str,
    metadata: dict | None = None,
):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        position=0,
        metadata=metadata or {},
    )


def test_bm25_returns_relevant_document():
    retriever = BM25Retriever(
        [
            chunk(
                "1",
                "OpenShift AI provides model serving",
            ),
            chunk(
                "2",
                "Gardening requires regular watering",
            ),
        ]
    )

    results = retriever.search(
        "OpenShift model serving",
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "1"
    assert results[0].score > 0


def test_bm25_applies_metadata_filter():
    retriever = BM25Retriever(
        [
            chunk(
                "1",
                "Granite embedding model",
                {"team": "ai"},
            ),
            chunk(
                "2",
                "Granite embedding deployment",
                {"team": "platform"},
            ),
        ]
    )

    results = retriever.search(
        "Granite embedding",
        metadata_filter=MetadataFilter(
            {
                "team": "platform",
            }
        ),
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "2"