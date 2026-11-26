import os
API_KEY = os.getenv("API_KEY", "redhat-demo-key")

MODEL_REGISTRY_FILE = os.getenv("MODEL_REGISTRY_FILE", "config/models.json")

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "5"))  # requests per minute
RATE_LIMIT_WINDOW_SECONDS= int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

# Redis-backed response cache . Empty -> in-memory fallback.
REDIS_URL = os.getenv("REDIS_URL", "")

# Authentication : comma-separated providers tried in order.
# Example: "oidc,api_key"
AUTH_PROVIDERS = [
    p.strip()
    for p in os.getenv("AUTH_PROVIDERS", "api_key").split(",")
    if p.strip()
]

# OIDC / Keycloak 
OIDC_ISSUER = os.getenv("OIDC_ISSUER", "")
OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE", "")
OIDC_JWKS_URL = os.getenv("OIDC_JWKS_URL", "")  # optional override, defaults to {issuer}/.well-known/jwks.json
OIDC_USERNAME_CLAIM = os.getenv("OIDC_USERNAME_CLAIM", "preferred_username")
OIDC_ROLE_CLAIM = os.getenv("OIDC_ROLE_CLAIM", "realm_access.roles")
OIDC_ADMIN_ROLES = {
    role.strip()
    for role in os.getenv("OIDC_ADMIN_ROLES", "admin,administrator,gateway-admin").split(",")
    if role.strip()
}

# API key store with roles and quotas 
API_KEYS_FILE = os.getenv("API_KEYS_FILE", "config/api_keys.json")

# Per-API-key quotas 
QUOTA_ENABLED = os.getenv("QUOTA_ENABLED", "true").lower() == "true"
DEFAULT_QUOTA_REQUESTS = int(os.getenv("DEFAULT_QUOTA_REQUESTS", "1000"))
QUOTA_WINDOW_SECONDS = int(os.getenv("QUOTA_WINDOW_SECONDS", "86400"))

# RAG Configuration
RAG_CONFIG_FILE = os.getenv("RAG_CONFIG_FILE", "config/rag_config.json")
RAG_DEFAULT_STRATEGY = os.getenv("RAG_DEFAULT_STRATEGY", "hybrid")
RAG_AUTO_STRATEGY = os.getenv("RAG_AUTO_STRATEGY", "false").lower() == "true"

# RAG reranking : "lexical" (no dependency) or "cross_encoder"
RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
RAG_RERANKER = os.getenv("RAG_RERANKER", "lexical")
CROSS_ENCODER_URL = os.getenv("CROSS_ENCODER_URL", "")  # e.g. TEI /rerank endpoint

# RAG feedback loop . Empty -> in-memory only.
FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", "")

# Tool calling / agents 
TOOLS_ENABLED = os.getenv("TOOLS_ENABLED", "true").lower() == "true"
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
AGENT_PLANNER_MODEL = os.getenv("AGENT_PLANNER_MODEL", "auto")
AGENT_EXECUTOR_MODEL = os.getenv("AGENT_EXECUTOR_MODEL", "auto")
AGENT_REVIEWER_MODEL = os.getenv("AGENT_REVIEWER_MODEL", "auto")

# Enterprise tool providers . Values come from a Secret in OpenShift.
KUBERNETES_API_URL = os.getenv("KUBERNETES_API_URL", "")  # empty -> in-cluster service account
GITHUB_API_URL = os.getenv("GITHUB_API_URL", "https://api.github.com")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
ARGOCD_URL = os.getenv("ARGOCD_URL", "")
ARGOCD_TOKEN = os.getenv("ARGOCD_TOKEN", "")

# MCP server 
MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() == "true"
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "goalway")
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")
