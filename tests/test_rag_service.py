from gateway.rag.chunker import TextChunker
from gateway.rag.embeddings import DummyEmbeddingProvider
from gateway.rag.models import Document
from gateway.rag.service import RAGService
from gateway.rag.vector_store import InMemoryVectorStore


def test_rag_service_ingest_and_search():
    service = RAGService(
        chunker=TextChunker(
            chunk_size=60,
            overlap=10,
        ),
        embedding_provider=DummyEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
    )

    count = service.ingest(
        Document(
            document_id="doc-1",
            text=(
                "OpenShift is a Kubernetes platform. "
                "RAG combines retrieval with generation. "
                "Ollama can serve local embedding models."
            ),
            metadata={"source": "test"},
        )
    )

    assert count > 0

    results, stats = service.search(
        query="local embedding model",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].chunk.document_id == "doc-1"
    assert stats is not None
    assert stats.returned_results == 2