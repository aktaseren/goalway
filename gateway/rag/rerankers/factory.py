from gateway.config import CROSS_ENCODER_URL, RAG_RERANKER
from gateway.rag.rerankers.base import Reranker
from gateway.rag.rerankers.cross_encoder import CrossEncoderReranker
from gateway.rag.rerankers.lexical import LexicalReranker


def create_reranker(name: str | None = None) -> Reranker:
    """
    Create a reranker by name.

    - "lexical": dependency-free second-stage scorer (default)
    - "cross_encoder": TEI /rerank endpoint (CROSS_ENCODER_URL),
      falls back to lexical when the endpoint is unavailable
    """
    reranker_name = (name or RAG_RERANKER).lower()

    if reranker_name == "cross_encoder":
        if not CROSS_ENCODER_URL:
            raise ValueError(
                "cross_encoder reranker requires CROSS_ENCODER_URL"
            )
        return CrossEncoderReranker(CROSS_ENCODER_URL)

    if reranker_name == "lexical":
        return LexicalReranker()

    raise ValueError(f"Unknown reranker: {reranker_name}")
