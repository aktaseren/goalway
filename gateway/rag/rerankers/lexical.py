"""
Dependency-free lexical reranker .

A second-stage relevance pass that scores each candidate chunk by
query-term coverage, term density and (as a mild prior) its
first-stage retrieval score. Not as accurate as a cross-encoder but
it needs no model, downloads or GPU, so it works everywhere --
including the default single-pod OpenShift deployment.
"""

import math
import re

from gateway.rag.rerankers.base import Reranker


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class LexicalReranker(Reranker):
    def __init__(self, first_stage_weight: float = 0.1):
        self.first_stage_weight = first_stage_weight

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5,
    ) -> list:
        query_terms = set(_tokenize(query))

        if not query_terms or not results:
            return results[:top_k]

        scored = []

        for result in results:
            chunk_text = result.chunk.text
            chunk_terms = _tokenize(chunk_text)

            if not chunk_terms:
                coverage = 0.0
                density = 0.0
            else:
                matches = sum(
                    1 for term in chunk_terms if term in query_terms
                )
                coverage = len(
                    query_terms & set(chunk_terms)
                ) / len(query_terms)
                density = matches / math.sqrt(len(chunk_terms))

            # Length normalization keeps long chunks from winning by
            # simply containing more text.
            lexical_score = coverage + 0.5 * density

            # Blend with the first-stage score so RRF signals survive.
            blended = (
                lexical_score
                + self.first_stage_weight * _normalize(result.score)
            )

            scored.append((blended, coverage, result))

        scored.sort(key=lambda item: item[0], reverse=True)

        reranked = []

        for blended, coverage, result in scored[:top_k]:
            # Carry the rerank score alongside the original retrieval
            # score so citations and stats can show both.
            reranked.append(
                result.model_copy(
                    update={
                        "rerank_score": round(blended, 4),
                        "reranker": "lexical",
                    }
                )
            )

        return reranked

    def name(self) -> str:
        return "lexical"


def _normalize(score: float) -> float:
    # RRF scores are ~0.03 or lower; cosine similarities are 0..1.
    # A soft normalization keeps both in a comparable range.
    if score <= 0:
        return 0.0

    return min(1.0, math.log10(1 + 100 * score))
