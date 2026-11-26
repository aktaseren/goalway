#!/bin/bash
# Basic RAG Usage Examples

API_URL="http://localhost:8080"
API_KEY="redhat-demo-key"

echo "=== Goalway - RAG Examples ==="
echo ""

# 1. Ingest Documents
echo "1. Ingesting documents..."
curl -X POST "$API_URL/v1/rag/ingest" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "document_id": "openshift-doc-1",
    "text": "OpenShift AI provides enterprise-grade model serving capabilities for machine learning models. It integrates with KServe and vLLM for scalable inference.",
    "metadata": {
      "team": "ai",
      "priority": 1,
      "category": "documentation",
      "created_at": "2024-01-15"
    }
  }'

echo -e "\n"

curl -X POST "$API_URL/v1/rag/ingest" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "document_id": "openshift-doc-2",
    "text": "Kubernetes orchestrates containerized applications across clusters. OpenShift builds on Kubernetes with additional enterprise features.",
    "metadata": {
      "team": "platform",
      "priority": 2,
      "category": "documentation",
      "created_at": "2024-02-20"
    }
  }'

echo -e "\n"

# 2. Basic Search (Default Hybrid Strategy)
echo "2. Basic search (hybrid strategy)..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "OpenShift model serving",
    "top_k": 3
  }' | jq '.'

echo -e "\n"

# 3. Search with Specific Strategy
echo "3. Search with vector strategy..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "What is model serving?",
    "top_k": 3,
    "strategy": "vector"
  }' | jq '.'

echo -e "\n"

# 4. Search with Metadata Filter
echo "4. Search with metadata filter..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "documentation",
    "top_k": 5,
    "metadata_filter": {
      "team": "ai"
    }
  }' | jq '.'

echo -e "\n"

# 5. Search with Statistics
echo "5. Search with statistics..."
curl -X POST "$API_URL/v1/rag/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "OpenShift Kubernetes",
    "top_k": 3,
    "include_stats": true
  }' | jq '.'

echo -e "\n"
echo "=== Examples Complete ==="
