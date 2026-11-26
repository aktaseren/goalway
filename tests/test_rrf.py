from gateway.rag.models import DocumentChunk
from gateway.rag.retrievers import (
    RetrievalResult,
    reciprocal_rank_fusion,
)


def result(
    chunk_id: str,
    rank: int,
    retriever: str,
):
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc",
            text=f"text-{chunk_id}",
            position=0,
            metadata={},
        ),
        score=1.0,
        rank=rank,
        retriever=retriever,
    )


def test_rrf_rewards_results_from_multiple_retrievers():
    lexical = [
        result("a", 1, "bm25"),
        result("b", 2, "bm25"),
    ]

    vector = [
        result("b", 1, "vector"),
        result("c", 2, "vector"),
    ]

    fused = reciprocal_rank_fusion(
        [
            lexical,
            vector,
        ],
        top_k=3,
    )

    assert fused[0].chunk.chunk_id == "b"
    assert fused[0].retriever == "hybrid"
    assert fused[0].metadata["sources"] == [
        "bm25",
        "vector",
    ]