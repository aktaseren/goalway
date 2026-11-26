from gateway.rag.models import DocumentChunk
from gateway.rag.retrievers import (
    HybridRetriever,
    RetrievalResult,
    RetrievalStrategy,
    Retriever,
)


class FakeRetriever(Retriever):
    def __init__(
        self,
        retriever_name: str,
        chunk_ids: list[str],
    ):
        self._name = retriever_name
        self._chunk_ids = chunk_ids

    @property
    def name(self) -> str:
        return self._name

    def search(
        self,
        query,
        top_k=5,
        metadata_filter=None,
    ):
        return [
            RetrievalResult(
                chunk=DocumentChunk(
                    chunk_id=chunk_id,
                    document_id="doc",
                    text=chunk_id,
                    position=0,
                    metadata={},
                ),
                score=1.0,
                rank=index,
                retriever=self.name,
            )
            for index, chunk_id in enumerate(
                self._chunk_ids[:top_k],
                start=1,
            )
        ]


def test_hybrid_retriever_fuses_results():
    retriever = HybridRetriever(
        lexical_retriever=FakeRetriever(
            "bm25",
            ["shared", "lexical"],
        ),
        vector_retriever=FakeRetriever(
            "vector",
            ["shared", "vector"],
        ),
    )

    results = retriever.search(
        "test query",
        top_k=3,
    )

    assert results[0].chunk.chunk_id == (
        "shared"
    )

    assert retriever.last_stats is not None
    assert (
        retriever.last_stats.strategy
        == RetrievalStrategy.HYBRID
    )