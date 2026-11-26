#!/bin/bash
# Advanced RAG Usage Examples

API_URL="http://localhost:8080"
API_KEY="redhat-demo-key"

echo "=== Advanced RAG Examples ==="
echo ""

# 1. Range Query with Metadata
echo "1. Range query (priority 1-3)..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "high priority documentation",
    "top_k": 10,
    "metadata_filter": {
      "priority": {
        "$gte": 1,
        "$lte": 3
      }
    }
  }' | jq '.results | length'

echo -e "\n"

# 2. Multi-Value Filter ($in operator)
echo "2. Multi-value filter (AI or ML team)..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "documentation",
    "top_k": 10,
    "metadata_filter": {
      "team": {
        "$in": ["ai", "ml", "data"]
      }
    }
  }' | jq '.results | length'

echo -e "\n"

# 3. Exclusion Filter ($nin operator)
echo "3. Exclusion filter (not archived)..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "documentation",
    "top_k": 10,
    "metadata_filter": {
      "status": {
        "$nin": ["archived", "deleted"]
      }
    }
  }' | jq '.results | length'

echo -e "\n"

# 4. Date Range Filter
echo "4. Date range filter (2024 Q1)..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "documentation",
    "top_k": 10,
    "metadata_filter": {
      "created_at": {
        "$gte": "2024-01-01",
        "$lt": "2024-04-01"
      }
    }
  }' | jq '.results | length'

echo -e "\n"

# 5. Complex Combined Filters
echo "5. Complex filter (team=ai AND priority<=2 AND not archived AND from 2024)..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "documentation",
    "top_k": 10,
    "strategy": "hybrid",
    "metadata_filter": {
      "team": "ai",
      "priority": {"$lte": 2},
      "status": {"$ne": "archived"},
      "created_at": {"$gte": "2024-01-01"}
    },
    "include_stats": true
  }' | jq '{
    query: .query,
    results: .results | length,
    stats: {
      strategy: .stats.strategy,
      total_latency_ms: .stats.total_latency_ms,
      overlap_percentage: .stats.overlap_percentage
    }
  }'

echo -e "\n"

# 6. Auto Strategy Selection
echo "6. Auto strategy selection..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "What is OpenShift?",
    "strategy": "auto",
    "include_stats": true
  }' | jq '{
    query: .query,
    selected_strategy: .stats.strategy
  }'

echo -e "\n"
echo "=== Advanced Examples Complete ==="
