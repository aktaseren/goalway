import time
from gateway.rag.retrievers.base import Retriever
from gateway.rag.retrievers.models import (
    MetadataFilter,
    RetrievalResult,
    RetrievalStats,
    RetrievalStrategy,
)
from gateway.rag.retrievers.rrf import (
    reciprocal_rank_fusion,
)


class HybridRetriever(Retriever):
    def __init__(
        self,
        lexical_retriever: Retriever,
        vector_retriever: Retriever,
        candidate_multiplier: int = 4,
        rank_constant: int = 60,
    ):
        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than zero"
            )

        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever
        self.candidate_multiplier = candidate_multiplier
        self.rank_constant = rank_constant
        self.last_stats: RetrievalStats | None = None

    @property
    def name(self) -> str:
        return "hybrid"

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        start_time = time.perf_counter()

        candidate_count = (
            top_k
            * self.candidate_multiplier
        )

        # Lexical retrieval with timing
        lexical_start = time.perf_counter()
        lexical_results = (
            self.lexical_retriever.search(
                query=query,
                top_k=candidate_count,
                metadata_filter=metadata_filter,
            )
        )
        lexical_time = (time.perf_counter() - lexical_start) * 1000

        # Vector retrieval with timing
        vector_start = time.perf_counter()
        vector_results = (
            self.vector_retriever.search(
                query=query,
                top_k=candidate_count,
                metadata_filter=metadata_filter,
            )
        )
        vector_time = (time.perf_counter() - vector_start) * 1000

        # Calculate overlap for recall estimation
        lexical_ids = {r.chunk.chunk_id for r in lexical_results}
        vector_ids = {r.chunk.chunk_id for r in vector_results}
        overlap_ids = lexical_ids & vector_ids
        overlap_count = len(overlap_ids)

        # Fusion with timing
        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [
                lexical_results,
                vector_results,
            ],
            top_k=top_k,
            rank_constant=self.rank_constant,
        )
        fusion_time = (time.perf_counter() - fusion_start) * 1000

        total_time = (time.perf_counter() - start_time) * 1000

        # Calculate overlap percentage in final results
        fused_ids = {r.chunk.chunk_id for r in fused}
        fused_from_both = len(overlap_ids & fused_ids)
        overlap_pct = (fused_from_both / len(fused) * 100) if fused else 0.0

        # Calculate RRF score statistics
        if fused:
            scores = [r.score for r in fused]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            avg_score = max_score = min_score = 0.0

        # Build comprehensive stats
        self.last_stats = (
            RetrievalStats(
                strategy=RetrievalStrategy.HYBRID,
                query=query,
                returned_results=len(fused),
                lexical_candidates=len(lexical_results),
                vector_candidates=len(vector_results),
            )
            .with_latency(
                lexical_ms=lexical_time,
                vector_ms=vector_time,
                fusion_ms=fusion_time,
                total_ms=total_time,
            )
            .with_recall(
                overlap=overlap_count,
                overlap_pct=overlap_pct,
            )
            .with_ranking(
                avg_score=avg_score,
                max_score=max_score,
                min_score=min_score,
            )
        )

        return fused