from collections import defaultdict

from gateway.rag.retrievers.models import (
    RetrievalResult,
)


def _chunk_key(result: RetrievalResult) -> str:
    chunk_id = getattr(
        result.chunk,
        "chunk_id",
        None,
    )

    if chunk_id is not None:
        return str(chunk_id)

    document_id = getattr(
        result.chunk,
        "document_id",
        "",
    )

    return (
        f"{document_id}:"
        f"{result.chunk.text}"
    )


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievalResult]],
    top_k: int = 5,
    rank_constant: int = 60,
) -> list[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if rank_constant <= 0:
        raise ValueError(
            "rank_constant must be greater than zero"
        )

    scores: dict[str, float] = defaultdict(float)
    results_by_key: dict[str, RetrievalResult] = {}
    sources: dict[str, set[str]] = defaultdict(set)

    for result_set in result_sets:
        for result in result_set:
            key = _chunk_key(result)

            scores[key] += 1.0 / (
                rank_constant
                + result.rank
            )

            results_by_key[key] = result
            sources[key].add(result.retriever)

    ordered_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    fused: list[RetrievalResult] = []

    for rank, key in enumerate(
        ordered_keys[:top_k],
        start=1,
    ):
        original = results_by_key[key]

        fused.append(
            RetrievalResult(
                chunk=original.chunk,
                score=scores[key],
                rank=rank,
                retriever="hybrid",
                metadata={
                    "sources": sorted(
                        sources[key]
                    ),
                },
            )
        )

    return fused