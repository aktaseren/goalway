#!/usr/bin/env bash
#
# End-to-end demo of the Goalway against a route URL,
# themed around football news.
#
#   ./examples/demo.sh                       # uses 'oc get route goalway'
#   ./examples/demo.sh http://goalway-... # explicit URL
#
# Runs: health -> seed football news -> RAG search with citations
# -> groundedness (with a fabricated claim) -> feedback -> evaluation
# -> MCP initialize/tools list/tools call -> chat completion (echo).
set -euo pipefail

API_KEY="${API_KEY:-redhat-demo-key}"
API_URL="${1:-}"

if [ -z "$API_URL" ]; then
  if command -v oc >/dev/null 2>&1; then
    HOST="$(oc get route goalway -o jsonpath='{.spec.host}' 2>/dev/null || true)"
    [ -n "$HOST" ] && API_URL="http://$HOST"
  fi
fi

if [ -z "$API_URL" ]; then
  echo "usage: $0 <gateway-url>" >&2
  echo "  (no argument given and 'oc get route goalway' failed)" >&2
  exit 1
fi

# pretty-print with python3 when available
if command -v python3 >/dev/null 2>&1; then
  PRETTY="python3 -m json.tool"
else
  PRETTY="cat"
fi

header() {
  printf '\e[1;36m\n=== %s ===\e[0m\n' "$1"
}

post() { # post PATH JSON
  curl -fsS -X POST "$API_URL$1" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$2" | $PRETTY
}

echo "Goalway — football news demo against $API_URL (key: $API_KEY)"

# 1. health ---------------------------------------------------------------
header "1. Health"
curl -fsS "$API_URL/health" | $PRETTY

# 2. seed the football knowledge base -------------------------------------
header "2. Seed knowledge base with football news"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/seed_football.py" "$API_URL" "$API_KEY"

# 2b. fetch live news from trusted sources --------------------------------
header "2b. Fetch live news (BBC Sport · Sky Sports · CNN Sport · TRT Spor)"
post "/v1/news/refresh" '{}'

# 3. RAG search with citations -------------------------------------------
header "3. RAG: search with citations + stats"
post "/v1/rag/search" '{
  "query": "who won the champions league final",
  "top_k": 3,
  "include_stats": true
}'

# 4. groundedness (one fabricated claim) ----------------------------------
header "4. Groundedness: answer with one fabricated claim"
post "/v1/rag/groundedness" '{
  "query": "who won the champions league finals",
  "answer": "Manchester City beat Inter Milan 1-0 to win the 2023 Champions League final. Real Madrid beat Borussia Dortmund 2-0 in the 2024 final. Barcelona won the 2024 Champions League final 7-0 against Bayern Munich, a match played on the moon."
}'

# 5. feedback ---------------------------------------------------------------
header "5. Feedback: rate the search results"
post "/v1/rag/feedback" '{
  "rating": 1,
  "query": "who won the champions league final",
  "document_id": "ucl-final-2023",
  "comment": "demo: relevant and cited"
}'

# 6. evaluation -------------------------------------------------------------
header "6. Evaluate: retrieval metrics for the query"
post "/v1/rag/evaluate" '{
  "query": "who won the champions league final",
  "relevant_document_ids": ["ucl-final-2023", "ucl-final-2024"],
  "top_k": 3,
  "k": 3
}'

# 7. MCP -------------------------------------------------------------------
header "7. MCP: initialize"
post "/mcp" '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}'

header "7. MCP: tools/list"
post "/mcp" '{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}'

header "7. MCP: tools/call rag_search"
post "/mcp" '{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "rag_search",
    "arguments": {"query": "messi transfer", "top_k": 2}
  }
}'

# 8. chat completions ------------------------------------------------------
header "8. Chat: enumeration — top 5 news (numbered, cited)"
post "/v1/chat/completions" '{
  "model": "auto",
  "messages": [{"role": "user", "content": "tell me top 5 todays sport news"}]
}'

header "8b. Chat: source-aware — turkish news (metadata-filtered retrieval)"
post "/v1/chat/completions" '{
  "model": "auto",
  "messages": [{"role": "user", "content": "tell me top 3 turkish football news"}]
}'

header "8c. Chat: recency — last champions league winner"
post "/v1/chat/completions" '{
  "model": "auto",
  "messages": [{"role": "user", "content": "who won the last champions league final"}]
}'

header "Demo complete"
echo "Interactive console: $API_URL/ui"
echo "Metrics:             $API_URL/metrics"
