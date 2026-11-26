import pytest

from gateway.rag.chunker import TextChunker
from gateway.rag.models import Document


def test_chunker_splits_text_with_overlap():
    document = Document(
        document_id="doc-1",
        text=(
            "OpenShift is a Kubernetes platform. "
            "It provides enterprise container orchestration. "
            "RAG combines retrieval with generation. "
            "Ollama serves local embedding models."
        ),
    )

    chunker = TextChunker(
        chunk_size=80,
        overlap=15,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == "doc-1" for chunk in chunks)


def test_chunker_handles_empty_document():
    chunker = TextChunker()

    chunks = chunker.chunk(
        Document(
            document_id="empty",
            text="   ",
        )
    )

    assert chunks == []


def test_chunker_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, overlap=100)