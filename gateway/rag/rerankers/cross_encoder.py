"""
Cross-encoder reranker .

Scores (query, chunk) pairs with a cross-encoder served behind an
OpenShift AI / KServe model-serving runtime that exposes the Hugging
Face Text Embeddings Inference (TEI) /rerank endpoint.

Configure with CROSS_ENCODER_URL, e.g.:

    CROSS_ENCODER_URL=http://reranker.goalway.svc.cluster.local:8080
"""

import logging

import requests

from gateway.rag.rerankers.base import Reranker
from gateway.rag.rerankers.lexical import LexicalReranker

logger = logging.getLogger("goalway.reranker")

REQUEST_TIMEOUT_SECONDS = 30


class CrossEncoderReranker(Reranker):
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.fallback = LexicalReranker()

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5,
    ) -> list:
        if not results:
            return results[:top_k]

        try:
            response = requests.post(
                f"{self.url}/rerank",
                json={
                    "query": query,
                    "texts": [result.chunk.text for result in results],
                    "top_k": top_k,
                    "return_text": False,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            ranking = response.json()

            reranked = []

            for item in ranking[:top_k]:
                result = results[item["index"]]
                reranked.append(
                    result.model_copy(
                        update={
                            "rerank_score": round(
                                float(item["score"]), 4
                            ),
                            "reranker": "cross_encoder",
                        }
                    )
                )

            return reranked

        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning(
                "cross-encoder rerank failed (%s), falling back to lexical",
                exc,
            )
            return self.fallback.rerank(query, results, top_k)

    def name(self) -> str:
        return "cross_encoder"
