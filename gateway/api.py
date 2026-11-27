import os
import time
import uuid
import logging
import requests
import json

from fastapi import Response, HTTPException, Header, Request, APIRouter, Depends

from gateway.models import ChatRequest
from gateway.config import (
    API_KEY, MODEL_REGISTRY_FILE, RATE_LIMIT, 
    RATE_LIMIT_WINDOW_SECONDS, CACHE_ENABLED, CACHE_TTL_SECONDS
)

from gateway.security import authenticate_api_key, is_prompt_safe
from gateway.ratelimit import check_rate_limit
from gateway.cache import make_cache_key, get_cached_response, store_cached_response
from gateway.registry import load_model_registry, get_model_config, list_models, refresh_health, is_model_healthy, MODEL_HEALTH
from gateway.metrics import (
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    metrics_response,
    MODEL_REQUESTS,
    RAG_SEARCH_REQUESTS,
    RAG_SEARCH_LATENCY,
    RAG_RESULTS_RETURNED,
    RAG_CANDIDATES,
    RAG_OVERLAP_PERCENTAGE,
    RAG_RRF_SCORES,
    RAG_RERANK_LATENCY,
    RAG_FEEDBACK,
    GROUNDEDNESS_SCORE,
    QUOTA_REJECTIONS,
)

from gateway.health import health_response

from gateway.router import forward_request

from gateway.streaming import stream_backend_response

from gateway.failover import get_failover_model

from gateway.loadbalancer import get_backend

from gateway.auth import get_auth_context
from gateway.auth_context import AuthContext

from gateway.rbac import require_permission

from gateway.conversations import (
    list_conversations,
    get_conversation,
    delete_conversation,
    create_conversation_id,
    add_messages,
    add_assistant_message,
)

from gateway.metrics import FAILOVERS

from prometheus_client import REGISTRY
from prometheus_client.openmetrics.exposition import generate_latest

from gateway.rag.chunker import TextChunker
from gateway.rag.embeddings import create_embedding_provider
from gateway.rag.models import (
    Document,
    RAGIngestRequest,
    RAGIngestResponse,
    RAGSearchItem,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGEvaluateRequest,
    RAGFeedbackRequest,
    GroundednessRequest,
    RetrievalStatsResponse,
)
from gateway.rag.service import RAGService
from gateway.rag.sqlite_vector_store import SQLiteVectorStore
from gateway.rag.retrievers import MetadataFilter, RetrievalStrategy
from gateway.rag.config import get_rag_config
from gateway.rag.strategy_selector import select_strategy

# RAG maturity
from gateway.rag.citations import build_citations
from gateway.rag.rerankers import create_reranker
from gateway.rag.evaluation import evaluate_retrieval, llm_judge
from gateway.rag.feedback import get_feedback_store
from gateway.rag.hallucination import groundedness_report
from gateway.rag.qa import has_recency_intent

# quotas, tools, agents, MCP
from gateway.config import (
    QUOTA_ENABLED,
    RAG_RERANK_ENABLED,
)
from gateway.keys import (
    QuotaExceededError,
    check_quota,
    list_keys,
    load_api_keys,
    quota_usage,
)
from gateway.model_selection import choose_model
from gateway.models import AgentRunRequest
from gateway.agents import Orchestrator
from gateway.mcp import mcp_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("goalway")

router = APIRouter()
rag_router = APIRouter()

rag_service = RAGService(
    chunker = TextChunker(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
        overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
    ),
    embedding_provider=create_embedding_provider(),
    vector_store=SQLiteVectorStore(
        os.getenv("RAG_VECTOR_DB_PATH", "vectors.db")
    ),
)

def choose_model(prompt:str) -> str:
    # moved to gateway.model_selection (kept as an alias for
    # backward compatibility with older tests)
    from gateway.model_selection import choose_model as _choose
    return _choose(prompt)

@router.get("/")
def root():
    return {
        "message": "Goalway running",
        "ui": "/ui",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "mcp": "/mcp",
    }

@router.get("/health")
def health():
    return health_response()

@router.post("/admin/reload")
def reload_registry(
    request: Request,
    x_api_key: str | None = Header(default=None),
    auth: AuthContext = Depends(get_auth_context),
):
    require_permission(
        auth,
        "admin.reload",
    )
    load_model_registry(MODEL_REGISTRY_FILE)
    model_health = refresh_health()

    # API keys are hot-reloadable too (ConfigMap-backed store)
    keys_loaded = load_api_keys()

    return {"status": "reloaded", 
            "models": list_models(),
            "model_health": model_health,
            "api_keys_loaded": keys_loaded,
    }

@router.get("/admin/conversations")
def admin_list_conversations(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    require_permission(auth, "admin.reload")

    return{
        "conversations": list_conversations()
    }

@router.get("/admin/conversations/{conversation_id}")
def admin_get_conversation(
    conversation_id: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    require_permission(auth, "admin.reload")

    conversation = get_conversation(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Conversation not found"},
        )
    
    return {
        "conversation_id": conversation_id,
        "messages": conversation,
    }

@router.delete("/admin/conversations/{conversation_id}")
def admin_delete_conversation(
    conversation_id: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
): 
    require_permission(auth, "admin.reload")

    deleted =delete_conversation(conversation_id)

    if deleted is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Conversation not found"},
        )
    
    return {
        "status": "deleted",
        "conversation_id": conversation_id,
    }

@router.get("/metrics")
def metrics():
    return Response(
        content = generate_latest(REGISTRY),
        media_type="application/openmetrics-text; version=1.0.0; charset=utf-8",
    )

@rag_router.post(
    "/v1/rag/ingest",
    response_model=RAGIngestResponse,
)
def rag_ingest(
    payload: RAGIngestRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    require_permission(
        auth,
        "rag.ingest",
    )

    document = Document(
        document_id = payload.document_id,
        text = payload.text,
        metadata = payload.metadata,
    )

    chunks_created =rag_service.ingest(document)

    return RAGIngestResponse(
        document_id = document.document_id,
        chunks_created=chunks_created,
    )

@rag_router.post(
    "/v1/rag/search",
    response_model=RAGSearchResponse,
)
def rag_search(
    payload: RAGSearchRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    require_permission(
        auth,
        "rag.search",
    )

    rag_config = get_rag_config()

    # Determine strategy (explicit request, auto-selection, or default)
    requested_strategy = RetrievalStrategy(payload.strategy) if payload.strategy != "auto" else None

    strategy = select_strategy(
        query=payload.query,
        requested_strategy=requested_strategy,
        auto_select=rag_config.auto_strategy_enabled,
        rules=rag_config.auto_strategy_rules,
    )

    # second-stage reranking.
    #
    # Retrieve a larger candidate pool and let the reranker cut it
    # down to top_k -- a reranker that only reorders the already-
    # truncated top-k can never promote better documents from
    # further down the candidate list.
    rerank_candidate_multiplier = int(
        os.getenv("RAG_RERANK_CANDIDATES_MULTIPLIER", "4")
    )

    retrieve_k = payload.top_k

    if payload.rerank and RAG_RERANK_ENABLED:
        retrieve_k = payload.top_k * rerank_candidate_multiplier

    results, stats = rag_service.search(
        query=payload.query,
        top_k=retrieve_k,
        strategy=strategy,
        metadata_filter=MetadataFilter(
            values=payload.metadata_filter
        ) if payload.metadata_filter else None,
    )

    reranker_name = None

    if payload.rerank and RAG_RERANK_ENABLED and results:
        reranker = create_reranker()
        reranker_name = reranker.name()

        rerank_start = time.time()
        results = reranker.rerank(
            query=payload.query,
            results=results,
            top_k=payload.top_k,
        )
        rerank_ms = (time.time() - rerank_start) * 1000

        RAG_RERANK_LATENCY.labels(reranker=reranker_name).observe(
            rerank_ms / 1000
        )

    # Recency-intent queries ("who won the LAST final") are ordered
    # newest-first among close-scoring results: lexical rankers have
    # no notion of time, and the densest older article would
    # otherwise outrank the most recent one.
    if results and has_recency_intent(payload.query):
        from gateway.rag.qa import _recency_order

        results = _recency_order(results)

    # Record Prometheus metrics
    if stats:
        strategy_name = stats.strategy.value

        RAG_SEARCH_REQUESTS.labels(strategy=strategy_name).inc()
        RAG_RESULTS_RETURNED.labels(strategy=strategy_name).observe(stats.returned_results)

        # Record latency metrics
        if stats.total_latency_ms > 0:
            RAG_SEARCH_LATENCY.labels(
                strategy=strategy_name,
                component="total"
            ).observe(stats.total_latency_ms / 1000)

        if stats.lexical_latency_ms > 0:
            RAG_SEARCH_LATENCY.labels(
                strategy=strategy_name,
                component="lexical"
            ).observe(stats.lexical_latency_ms / 1000)

        if stats.vector_latency_ms > 0:
            RAG_SEARCH_LATENCY.labels(
                strategy=strategy_name,
                component="vector"
            ).observe(stats.vector_latency_ms / 1000)

        if stats.fusion_latency_ms > 0:
            RAG_SEARCH_LATENCY.labels(
                strategy=strategy_name,
                component="fusion"
            ).observe(stats.fusion_latency_ms / 1000)

        # Record candidate counts
        if stats.lexical_candidates > 0:
            RAG_CANDIDATES.labels(retriever_type="lexical").observe(stats.lexical_candidates)

        if stats.vector_candidates > 0:
            RAG_CANDIDATES.labels(retriever_type="vector").observe(stats.vector_candidates)

        # Record overlap percentage
        if stats.overlap_percentage > 0:
            RAG_OVERLAP_PERCENTAGE.observe(stats.overlap_percentage)

        # Record RRF scores
        if stats.avg_rrf_score > 0:
            RAG_RRF_SCORES.labels(score_type="avg").observe(stats.avg_rrf_score)
            RAG_RRF_SCORES.labels(score_type="max").observe(stats.max_rrf_score)
            RAG_RRF_SCORES.labels(score_type="min").observe(stats.min_rrf_score)

    stats_response = None
    if payload.include_stats and stats:
        stats_response = RetrievalStatsResponse(
            strategy=stats.strategy.value,
            returned_results=stats.returned_results,
            lexical_candidates=stats.lexical_candidates,
            vector_candidates=stats.vector_candidates,
            lexical_latency_ms=stats.lexical_latency_ms,
            vector_latency_ms=stats.vector_latency_ms,
            fusion_latency_ms=stats.fusion_latency_ms,
            total_latency_ms=stats.total_latency_ms,
            lexical_vector_overlap=stats.lexical_vector_overlap,
            overlap_percentage=stats.overlap_percentage,
            avg_rrf_score=stats.avg_rrf_score,
            max_rrf_score=stats.max_rrf_score,
            min_rrf_score=stats.min_rrf_score,
        )

    # citations with source spans
    citations = None

    if payload.include_citations and results:
        citations = build_citations(results, payload.query)

    search_id = f"search-{uuid.uuid4().hex[:12]}"

    return RAGSearchResponse(
        query=payload.query,
        results=[
            RAGSearchItem(
                chunk_id=result.chunk.chunk_id,
                document_id=result.chunk.document_id,
                text=result.chunk.text,
                position=result.chunk.position,
                metadata=result.chunk.metadata,
                score=result.score,
                rerank_score=result.rerank_score,
                reranker=result.reranker,
            )
            for result in results
        ],
        stats=stats_response,
        citations=citations,
        search_id=search_id,
    )

@rag_router.post("/v1/rag/evaluate")
def rag_evaluate(
    payload: RAGEvaluateRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """evaluation harness -- retrieval metrics + LLM-as-judge."""
    require_permission(auth, "rag.evaluate")

    # Measure the same pipeline /v1/rag/search serves: retrieve a
    # candidate pool, rerank, cut to top_k.
    retrieve_k = payload.top_k

    if RAG_RERANK_ENABLED:
        retrieve_k = payload.top_k * int(
            os.getenv("RAG_RERANK_CANDIDATES_MULTIPLIER", "4")
        )

    results, _ = rag_service.search(
        query=payload.query,
        top_k=retrieve_k,
    )

    if RAG_RERANK_ENABLED and results:
        results = create_reranker().rerank(
            query=payload.query,
            results=results,
            top_k=payload.top_k,
        )

    evaluation = {
        "query": payload.query,
        "retrieval": evaluate_retrieval(
            payload.query,
            results,
            payload.relevant_document_ids,
            k=payload.k,
        ),
    }

    if payload.answer:
        evaluation["answer"] = llm_judge(
            payload.query,
            payload.answer,
            results,
            model=payload.model,
        )

    return evaluation


@rag_router.post("/v1/rag/feedback")
def rag_feedback(
    payload: RAGFeedbackRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """feedback loop -- record a rating for search results."""
    require_permission(auth, "rag.feedback")

    if payload.rating == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "rating must be 1 (positive) or -1 (negative)"},
        )

    entry = get_feedback_store().record(
        rating=payload.rating,
        query=payload.query,
        document_id=payload.document_id,
        comment=payload.comment,
        username=auth.username,
        search_id=payload.search_id,
    )

    RAG_FEEDBACK.labels(
        rating="positive" if payload.rating == 1 else "negative"
    ).inc()

    return {
        "status": "recorded",
        "feedback_id": entry["feedback_id"],
    }


@rag_router.get("/v1/rag/feedback/stats")
def rag_feedback_stats(
    auth: AuthContext = Depends(get_auth_context),
):
    """feedback loop -- aggregated satisfaction stats."""
    require_permission(auth, "rag.search")

    return get_feedback_store().stats()


@rag_router.post("/v1/rag/groundedness")
def rag_groundedness(
    payload: GroundednessRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    hallucination detection.

    Checks an answer for groundedness either against retrieved
    context for the query (when given) or against a provided answer
    alone (returns unsupported claims against an empty context).
    """
    require_permission(auth, "rag.search")

    if payload.query:
        results, _ = rag_service.search(
            query=payload.query,
            top_k=payload.top_k,
        )

        report = groundedness_report(payload.answer, results)
        report["documents_checked"] = [
            result.chunk.document_id for result in results
        ]
    else:
        report = groundedness_report(payload.answer, [])
        report["documents_checked"] = []

    GROUNDEDNESS_SCORE.observe(report["groundedness_score"])

    return report


@router.post("/v1/agents/run")
def agents_run(
    payload: AgentRunRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """multi-agent orchestration (planner/researcher/executor/reviewer)."""
    require_permission(auth, "agents.run")

    orchestrator = Orchestrator(
        planner_model=payload.planner_model or "",
        executor_model=payload.executor_model or "",
        reviewer_model=payload.reviewer_model or "",
        use_rag=payload.use_rag,
    )

    try:
        result = orchestrator.run(
            objective=payload.objective,
            max_steps=payload.max_steps,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Agent run failed",
                "details": str(exc),
            },
        )

    return result.to_dict()


@router.get("/admin/keys")
def admin_list_keys(
    auth: AuthContext = Depends(get_auth_context),
):
    """list API keys (metadata only, no secrets)."""
    require_permission(auth, "admin.reload")

    return {"keys": list_keys()}


@router.get("/admin/quota")
def admin_quota(
    auth: AuthContext = Depends(get_auth_context),
):
    """per-key quota usage."""
    require_permission(auth, "admin.reload")

    return {"usage": quota_usage()}


def rag_chat_completion(
    req,
    prompt: str,
    model: str,
    request: Request,
    failover_from: str | None = None,
):
    """
    chat completion answered from the RAG knowledge base.

    Retrieval + rerank + extractive answering with citations — the
    same pipeline /v1/rag/search serves, surfaced through the
    OpenAI-compatible chat shape. Degrades to an honest "not in the
    knowledge base" reply when nothing relevant is found.
    """
    from gateway.rag.qa import build_extractive_answer
    from gateway.rag.rerankers import create_reranker

    request_id = str(uuid.uuid4())
    conversation_id = req.conversation_id or create_conversation_id()

    start = time.time()

    retrieve_k = 4 * int(
        os.getenv("RAG_RERANK_CANDIDATES_MULTIPLIER", "4")
    )

    results, _ = rag_service.search(query=prompt, top_k=retrieve_k)

    if RAG_RERANK_ENABLED and results:
        results = create_reranker().rerank(
            query=prompt,
            results=results,
            top_k=3,
        )

    # Relevance gate: when the top result doesn't actually cover the
    # question's content terms, say so instead of quoting
    # marginally-related text. Rerank scores alone can't separate
    # these — stop-word overlap inflates them.
    from gateway.rag.qa import content_coverage

    min_coverage = float(os.getenv("RAG_MIN_ANSWER_COVERAGE", "0.25"))

    top_score = 0.0
    top_coverage = 0.0
    answer_coverage = 0.0

    if results:
        top_score = (
            results[0].rerank_score
            if results[0].rerank_score is not None
            else results[0].score
        )
        top_coverage = content_coverage(prompt, results[0].chunk.text)

    if not results or top_coverage < min_coverage:
        horizon = _knowledge_horizon(results)

        answer = (
            "I don't have anything about that in my knowledge base"
            + (f" (news through {horizon})" if horizon else "")
            + ". I can answer questions about the ingested football "
            "news — or ingest more documents in the RAG tab."
        )

        results = []
        citations = []
    else:
        answer, citations = build_extractive_answer(
            prompt,
            results,
            fetch_document=rag_service.get_document,
        )

        # Second gate, on the answer itself: the chunk may mention
        # the question's terms somewhere without any sentence
        # actually answering it ("names of all squad members" vs a
        # transfer article that merely names the club). Term overlap
        # is not answerability — decline instead of misquoting.
        min_answer_coverage = float(
            os.getenv("RAG_MIN_ANSWER_COVERAGE", "0.5")
        )

        answer_body = answer.split("\n\nSources:")[0]
        answer_coverage = content_coverage(prompt, answer_body)

        if answer_coverage < min_answer_coverage:
            horizon = _knowledge_horizon(results)

            answer = (
                "I don't have anything about that in my knowledge base"
                + (f" (news through {horizon})" if horizon else "")
                + ". I can answer questions about the ingested football "
                "news — or ingest more documents in the RAG tab."
            )

            logger.info(
                "request_id=%s rag answer declined: chunk coverage %.2f "
                "but answer coverage %.2f",
                request_id,
                top_coverage,
                answer_coverage,
            )

            results = []
            citations = []
        else:
            horizon = _knowledge_horizon(results)

            if horizon:
                answer += f"\n\n(Knowledge base current through {horizon}.)"

    add_messages(conversation_id, req.messages)
    add_assistant_message(conversation_id, answer)

    MODEL_REQUESTS.labels(model=model).inc()
    REQUEST_COUNTER.labels(model=model).inc()

    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(
        "request_id=%s model=rag conversation_id=%s documents=%s top_score=%.2f",
        request_id,
        conversation_id,
        [result.chunk.document_id for result in results],
        top_score,
    )

    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "gateway": {
            "request_id": request_id,
            "conversation_id": conversation_id,
            "model": model,
            "failover_from": failover_from,
            "latency_ms": latency_ms,
            "documents": [
                result.chunk.document_id for result in results[:3]
            ],
            "citations": citations[:3],
            "top_score": round(top_score, 4),
            "top_coverage": round(top_coverage, 4),
            "answer_coverage": round(
                answer_coverage if results else 0.0, 4
            ),
            "cache": "miss",
        },
    }


def _knowledge_horizon(results: list) -> int | None:
    """Latest year across result metadata — the KB's freshness."""
    years = [
        result.chunk.metadata.get("year")
        for result in results
        if result.chunk.metadata.get("year")
    ]

    return max(years) if years else None


@router.post("/v1/chat/completions")
def chat(
    req: ChatRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
    auth: AuthContext = Depends(get_auth_context),
):
    require_permission(
        auth,
        "chat.complete",
    )
    client_key = f"{x_api_key}:{request.client.host}"
    check_rate_limit(client_key, RATE_LIMIT, RATE_LIMIT_WINDOW_SECONDS)

    # per-API-key quotas (distinct from per-minute rate limits)
    if QUOTA_ENABLED and auth.auth_type == "api_key" and auth.api_key:
        try:
            check_quota(auth.api_key)
        except QuotaExceededError as exc:
            QUOTA_REJECTIONS.inc()

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Quota exceeded",
                    "key": exc.key_name,
                    "limit": exc.limit,
                    "retry_after_seconds": exc.reset_in_seconds,
                },
            )
    
    prompt = req.messages[-1].get("content", "")

    if not is_prompt_safe(prompt):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Prompt blocked by AI Gateway policy",
                "reason": "Potential prompt injection detected"
            },
        )

    requested_model = req.model.lower()

    if requested_model == "auto":
        model = choose_model(prompt)
    else:
        model = requested_model

    # agentic tool-calling loop. When the client sends tools,
    # run the completion loop through the gateway tool registry.
    if req.tools:
        from gateway.config import TOOLS_ENABLED
        from gateway.llm import LLMError, complete_with_tools

        if not TOOLS_ENABLED:
            raise HTTPException(
                status_code=400,
                detail={"error": "Tool calling is disabled on this gateway"},
            )

        try:
            tool_result = complete_with_tools(
                model=model,
                messages=req.messages,
                tools=req.tools,
            )
        except LLMError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Tool-calling completion failed",
                    "details": str(exc),
                },
            )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": tool_result["answer"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "gateway": {
                "model": model,
                "tool_calls": tool_result["tool_calls"],
                "iterations": tool_result["iterations"],
                "stopped_early": tool_result.get("stopped_early", False),
            },
        }

    model_config = get_model_config(model)

    if not model_config or not model_config.get("enabled", False):
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown or disabled model: {req.model}"}
        )

    failover_from = None

    if not is_model_healthy(model):
        failover_model = get_failover_model(model)

        if not failover_model:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": f"Backend model service is unhealthy: {req.model}",
                    "failover": "no healthy fallback available",
                },
            )
        failover_from = model
        FAILOVERS.inc()
        model = failover_model
        model_config = get_model_config(model)

    # built-in RAG model — answers from the knowledge base via
    # retrieval + extraction, no backend service needed. Checked
    # AFTER failover so an unhealthy backend degrades to knowledge-
    # base answers instead of erroring.
    if model_config.get("provider") == "rag":
        return rag_chat_completion(
            req, prompt, model, request, failover_from=failover_from
        )

    cache_key = make_cache_key(model, req.messages)

    if CACHE_ENABLED:
        cached_data = get_cached_response(cache_key, CACHE_TTL_SECONDS)
        if cached_data:
            return cached_data

    backends = model_config.get("backends") or [model_config.get("url")]
    backend_url = get_backend(model, backends)

    if not backend_url:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Model: '{model}' has no backend URL configured."}
        )

    request_id = str(uuid.uuid4())
    
    logger.info({
                "event": "request_received",
                "request_id": request_id,
                "model": model,
                "backend_url": backend_url,
                "client_ip": request.client.host,
        })

    return forward_request(
        provider_name=model_config.get("provider", "openai_compatible"),
        req=req, 
        request=request, 
        model=model, 
        backend_url=backend_url,
        failover_from=failover_from,
    )

@router.post("/v1/chat/completions/stream")
def stream_chat(
    req: ChatRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
    auth: AuthContext = Depends(get_auth_context),
):
    require_permission(
        auth,
        "chat.complete"
    )
    prompt = req.messages[-1].get("content", "")

    if not is_prompt_safe(prompt):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Prompt blocked by AI Gateway policy",
                "reason": "Potential prompt injection detected"
            },
        )

    requested_model = req.model.lower()

    if requested_model == "auto":
        model = choose_model(prompt)
    else:
        model = requested_model

    model_config = get_model_config(model)

    if not model_config or not model_config.get("enabled", False):
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown or disabled model: {req.model}"}
        )
    
    if not is_model_healthy(model):
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"Backend model service is unhealthy",
                "model": model,
            },
        )

    return stream_backend_response(
        backend_url = model_config["url"],
        payload = req.model_dump(),
    )