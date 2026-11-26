"""Tests for the built-in RAG chat model ."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gateway.api as api_module
from gateway.main import app
from gateway.rag.models import Document, DocumentChunk, SearchResult
from gateway.rag.qa import build_extractive_answer

from fastapi.testclient import TestClient

client = TestClient(app)

UCL_TEXT = (
    "Real Madrid won the 2024 Champions League final, beating "
    "Borussia Dortmund 2-0 at Wembley. It was Real Madrid's 15th "
    "European Cup."
)


class FakeRAGService:
    def search(self, query, top_k=5, **kwargs):
        return (
            [
                SearchResult(
                    chunk=DocumentChunk(
                        chunk_id="ucl-2024-0",
                        document_id="ucl-final-2024",
                        text=UCL_TEXT,
                        position=0,
                        metadata={"year": 2024},
                    ),
                    score=0.9,
                )
            ],
            None,
        )


def _fake_rag(monkeypatch):
    monkeypatch.setattr(api_module, "rag_service", FakeRAGService())


def test_extractive_answer_quotes_source_with_citation():
    results = FakeRAGService().search("who won the champions league")[0]

    answer, citations = build_extractive_answer(
        "who won the champions league final", results
    )

    assert "Real Madrid won the 2024 Champions League final" in answer
    assert "[1]" in answer
    assert "Sources: [1] ucl-final-2024" in answer
    assert citations[0]["document_id"] == "ucl-final-2024"
    assert citations[0]["spans"]


def test_extractive_answer_empty_knowledge_base():
    answer, citations = build_extractive_answer("anything", [])

    assert "knowledge base" in answer.lower()
    assert citations == []


def test_recency_ordering_leads_with_newest_close_score():
    """'Last champion' questions: close scores -> newest year first."""
    def result(doc_id, year, score, with_year=True):
        metadata = {"year": year} if with_year else {}

        return SearchResult(
            chunk=DocumentChunk(
                chunk_id=f"{doc_id}-0",
                document_id=doc_id,
                text=f"Team won the final in {year}.",
                position=0,
                metadata=metadata,
            ),
            score=score,
            rerank_score=score,
        )

    # Mirrors the real cluster scores for this query: 2023 doc is
    # lexically stronger, but the 2024 doc is within 2x relevance.
    results = [
        result("ucl-2023", 2023, 1.10),
        result("var", 2024, 0.84, with_year=False),
        result("ucl-2024", 2024, 0.62),
    ]

    answer, citations = build_extractive_answer(
        "who is the last champion", results
    )

    # Newest close-scoring result leads the answer...
    assert answer.index("2024") < answer.index("2023")

    # ...and citations stay aligned with the quoted documents.
    assert citations[0]["document_id"] == "ucl-2024"
    assert citations[1]["document_id"] == "ucl-2023"
    assert answer.index("ucl-2024") < answer.index("ucl-2023")


def test_rag_model_registered_and_healthy():
    import gateway.registry as registry

    registry.refresh_health()

    assert "rag" in registry.list_models()
    assert registry.is_model_healthy("rag") is True


def test_chat_with_rag_model_answers_with_citations(monkeypatch):
    _fake_rag(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "rag",
            "messages": [
                {"role": "user", "content": "who won the champions league final"}
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    content = data["choices"][0]["message"]["content"]
    assert "Real Madrid" in content
    assert "[1]" in content

    gateway = data["gateway"]
    assert gateway["model"] == "rag"
    assert gateway["documents"] == ["ucl-final-2024"]
    assert gateway["citations"][0]["document_id"] == "ucl-final-2024"


def test_auto_routes_football_question_to_rag_chat(monkeypatch):
    """model=auto + football question -> rag model -> cited answer."""
    _fake_rag(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "tell me who is last champion in champions league",
                }
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["gateway"]["model"] == "rag"
    assert "Real Madrid" in data["choices"][0]["message"]["content"]


def test_rag_chat_says_no_when_nothing_matches(monkeypatch):
    """Questions outside the knowledge base get an honest no-answer
    with the KB horizon instead of marginally-related quotes."""
    _fake_rag(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "rag",
            "messages": [
                {"role": "user", "content": "what is the current date"}
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    content = data["choices"][0]["message"]["content"]
    assert "don't have anything about that" in content
    # "current" / "date" never appear in the UCL text -> coverage 0
    assert data["gateway"]["top_coverage"] == 0.0
    assert data["gateway"]["citations"] == []


def test_rag_chat_discloses_knowledge_horizon(monkeypatch):
    _fake_rag(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "rag",
            "messages": [
                {"role": "user", "content": "who won the champions league"}
            ],
        },
    )

    data = response.json()

    assert "(Knowledge base current through" in (
        data["choices"][0]["message"]["content"]
    )


def test_unhealthy_backend_fails_over_to_rag(monkeypatch):
    """model=auto + non-football question -> tinyllama (unhealthy) ->
    failover lands on the rag model and answers instead of erroring
    with 'no backend URL configured'."""
    _fake_rag(monkeypatch)

    import gateway.registry as registry

    # tinyllama is unhealthy in this environment (no backend), and
    # rag is the only healthy model -> failover target.
    registry.refresh_health()

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "what is the current date"}
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["gateway"]["model"] == "rag"
    assert data["gateway"]["failover_from"] == "tinyllama"
    assert "don't have anything about that" in (
        data["choices"][0]["message"]["content"]
    )


def test_ingest_is_idempotent(monkeypatch, tmp_path):
    """Re-ingesting a document replaces it, never duplicates."""
    from gateway.rag.service import RAGService
    from gateway.rag.sqlite_vector_store import SQLiteVectorStore
    from gateway.rag.embeddings.dummy import DummyEmbeddingProvider

    store = SQLiteVectorStore(str(tmp_path / "vectors.db"))
    service = RAGService(
        embedding_provider=DummyEmbeddingProvider(),
        vector_store=store,
    )

    doc = Document(document_id="doc-1", text="First version of the text.")

    service.ingest(doc)
    service.ingest(doc)
    service.ingest(
        Document(document_id="doc-1", text="Second version, one chunk.")
    )

    assert len(store.all_chunks()) == 1
    assert len(service._chunks_cache) == 1
    assert service._chunks_cache[0].text == "Second version, one chunk."
