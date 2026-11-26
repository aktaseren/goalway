from types import SimpleNamespace

import gateway.api as api_module
from app import app
from fastapi.testclient import TestClient
from gateway.auth import get_auth_context
from gateway.rag.models import DocumentChunk, SearchResult
from gateway.rag.retrievers import RetrievalStrategy, RetrievalStats
client = TestClient(app)


class FakeRAGService:
    def ingest(self, document):
        return 2

    def search(self, query: str, top_k: int = 5, strategy=None, metadata_filter=None):
        results = [
            SearchResult(
                chunk=DocumentChunk(
                    chunk_id="doc-1-0",
                    document_id="doc-1",
                    text="OpenShift AI provides enterprise model serving.",
                    position=0,
                    metadata={"source": "test"},
                ),
                score=0.95,
            )
        ][:top_k]

        stats = RetrievalStats(
            strategy=strategy or RetrievalStrategy.HYBRID,
            query=query,
            returned_results=len(results),
            lexical_candidates=1,
            vector_candidates=1,
        )

        return results, stats


def set_role(role: str):
    app.dependency_overrides[get_auth_context] = lambda: SimpleNamespace(
        role=role
    )


def teardown_function():
    app.dependency_overrides.clear()


def test_admin_can_ingest_document(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "rag_service",
        FakeRAGService(),
    )
    set_role("admin")

    response = client.post(
        "/v1/rag/ingest",
        json={
            "document_id": "doc-1",
            "text": "OpenShift AI provides enterprise model serving.",
            "metadata": {"source": "test"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-1",
        "chunks_created": 2,
        "status": "ingested",
    }


def test_user_can_search_documents(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "rag_service",
        FakeRAGService(),
    )
    set_role("user")

    response = client.post(
        "/v1/rag/search",
        json={
            "query": "What provides enterprise model serving?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "What provides enterprise model serving?"
    assert len(body["results"]) == 1
    assert body["results"][0]["document_id"] == "doc-1"
    assert body["results"][0]["score"] == 0.95


def test_user_cannot_ingest_document(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "rag_service",
        FakeRAGService(),
    )
    set_role("user")

    response = client.post(
        "/v1/rag/ingest",
        json={
            "document_id": "doc-1",
            "text": "OpenShift AI provides enterprise model serving.",
            "metadata": {},
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"]["permission"] == "rag.ingest"

def test_rag_ingest_requires_permission():
    response = client.post(
        "/v1/rag/ingest",
        json={
            "document_id": "doc-1",
            "text": "OpenShift AI provides enterprise model serving.",
            "metadata": {"source": "test"},
        }
    )

    assert response.status_code in {401, 403}

def test_rag_search_requires_permission():
    response = client.post(
        "/v1/rag/search",
        json={
            "query": "What provides enterprise model serving?",
            "top_k": 3,
        },
    )

    assert response.status_code in {401, 403}