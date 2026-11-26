from typing import Any
from pydantic import BaseModel, Field

class Document(BaseModel):
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    position: int
    metadata: dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    rerank_score: float | None = None
    reranker: str | None = None

class RAGIngestRequest(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

class RAGIngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    status: str = "ingested"

class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    strategy: str = "hybrid"
    metadata_filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filter supporting operators ($eq, $ne, $lt, $lte, $gt, $gte, $in, $nin)"
    )
    include_stats: bool = False

    
    rerank: bool = True
    include_citations: bool = True

class RAGSearchItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    position: int
    metadata: dict[str, Any]
    score: float

    
    rerank_score: float | None = None
    reranker: str | None = None

class RetrievalStatsResponse(BaseModel):
    strategy: str
    returned_results: int
    lexical_candidates: int
    vector_candidates: int

    # Latency tracking (milliseconds)
    lexical_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Recall estimation
    lexical_vector_overlap: int = 0
    overlap_percentage: float = 0.0

    # Ranking diagnostics
    avg_rrf_score: float = 0.0
    max_rrf_score: float = 0.0
    min_rrf_score: float = 0.0

class RAGSearchResponse(BaseModel):
    query: str
    results: list[RAGSearchItem]
    stats: RetrievalStatsResponse | None = None

    
    citations: list[dict[str, Any]] | None = None
    search_id: str | None = None


# ---------------------------------------------------------------
# evaluation, feedback, groundedness
# ---------------------------------------------------------------

class RAGEvaluateRequest(BaseModel):
    query: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    k: int = Field(default=5, ge=1, le=50)
    answer: str | None = Field(
        default=None,
        description="Optional answer to also evaluate with the LLM judge",
    )
    model: str = "auto"


class RAGFeedbackRequest(BaseModel):
    rating: int = Field(ge=-1, le=1)
    query: str = ""
    document_id: str = ""
    comment: str = ""
    search_id: str = ""


class GroundednessRequest(BaseModel):
    query: str = ""
    answer: str = Field(min_length=1)
    # Optional: search first and check groundedness against results
    top_k: int = Field(default=5, ge=1, le=20)