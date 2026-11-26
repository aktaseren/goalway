from prometheus_client import Counter, Gauge, Histogram, generate_latest
from fastapi.responses import Response

REQUEST_COUNTER = Counter(
    "gateway_requests_total",
    "Total gateway requests",
    ["model"]
)  

REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "Gateway request latency in seconds",
    ["model"]
)

MODEL_REQUESTS = Counter(
    "gateway_model_requests_total",
    "Requests per model",
    ["model"],
)

CACHE_HITS = Counter(
    "gateway_cache_hits_total",
    "Cache hits",
)

CACHE_MISSES = Counter(
    "gateway_cache_misses_total",
    "Cache misses",
)

CACHE_ERRORS = Counter(
    "gateway_cache_errors_total",
    "Cache backend failures (degraded to miss / skipped store)",
)

FAILOVERS = Counter(
    "gateway_failovers_total",
    "Failover events",
)

RETRIES = Counter(
    "gateway_retries_total",
    "Retry attempts",
)

ACTIVE_CONVERSATIONS = Gauge(
    "gateway_active_conversations",
    "Current active conversation",
)

# RAG Metrics
RAG_SEARCH_REQUESTS = Counter(
    "rag_search_requests_total",
    "Total RAG search requests",
    ["strategy"],
)

RAG_SEARCH_LATENCY = Histogram(
    "rag_search_latency_seconds",
    "RAG search latency in seconds",
    ["strategy", "component"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RAG_RESULTS_RETURNED = Histogram(
    "rag_results_returned",
    "Number of results returned from RAG search",
    ["strategy"],
    buckets=[1, 2, 5, 10, 20, 50, 100],
)

RAG_CANDIDATES = Histogram(
    "rag_candidates_retrieved",
    "Number of candidates retrieved before fusion",
    ["retriever_type"],
    buckets=[1, 5, 10, 20, 50, 100, 200],
)

RAG_OVERLAP_PERCENTAGE = Histogram(
    "rag_overlap_percentage",
    "Percentage of results from both lexical and vector",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

RAG_RRF_SCORES = Histogram(
    "rag_rrf_scores",
    "RRF scores for hybrid retrieval",
    ["score_type"],  # avg, max, min
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# metrics --------------------------------------------------------

RAG_RERANK_LATENCY = Histogram(
    "rag_rerank_latency_seconds",
    "RAG rerank latency in seconds",
    ["reranker"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

RAG_FEEDBACK = Counter(
    "rag_feedback_total",
    "User feedback events",
    ["rating"],  # positive / negative
)

GROUNDEDNESS_SCORE = Histogram(
    "rag_groundedness_score",
    "Groundedness score of evaluated answers",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
)

TOOL_CALLS = Counter(
    "gateway_tool_calls_total",
    "Tool invocations",
    ["tool", "source"],  # source: chat | mcp | agent
)

MCP_REQUESTS = Counter(
    "mcp_requests_total",
    "MCP server requests",
)

AGENT_RUNS = Counter(
    "agent_runs_total",
    "Multi-agent orchestration runs",
)

AGENT_STEP_LATENCY = Histogram(
    "agent_step_latency_seconds",
    "Latency per agent role step",
    ["agent"],  # planner | researcher | executor | reviewer
    buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
)

QUOTA_REJECTIONS = Counter(
    "gateway_quota_rejections_total",
    "Requests rejected for exceeding per-key quotas",
)

def metrics_response():
    return Response(generate_latest(), media_type="text/plain")