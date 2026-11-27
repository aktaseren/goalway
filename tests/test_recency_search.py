"""Tests for recency-intent-aware search ordering."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from gateway.rag.qa import has_recency_intent, _recency_order
from gateway.rag.models import DocumentChunk, SearchResult


@pytest.fixture
def client_setup(monkeypatch):
    from fastapi.testclient import TestClient

    from gateway.main import app

    return TestClient(app), monkeypatch


def result(doc_id, year, score):
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=f"{doc_id}-0",
            document_id=doc_id,
            text=f"Team won the champions league final in {year}.",
            position=0,
            metadata={"year": year},
        ),
        score=score,
        rerank_score=score,
    )


def test_recency_intent_detection():
    assert has_recency_intent("who won the last champions league final")
    assert has_recency_intent("latest transfer news")
    assert has_recency_intent("the current champion")
    assert not has_recency_intent("who won the 2023 champions league final")
    assert not has_recency_intent("messi transfer")


def test_recency_order_newest_first_within_close_scores():
    results = [
        result("ucl-2023", 2023, 1.59),
        result("ucl-2025", 2025, 1.42),
        result("ucl-2024", 2024, 1.38),
    ]

    ordered = _recency_order(results)

    assert [r.chunk.document_id for r in ordered] == [
        "ucl-2025",
        "ucl-2024",
        "ucl-2023",
    ]


def test_recency_order_respects_clear_relevance_gaps():
    # 2025 doc scores less than half the best -> stays behind
    results = [
        result("strong-old", 2023, 1.60),
        result("weak-new", 2025, 0.70),
    ]

    ordered = _recency_order(results)

    assert ordered[0].chunk.document_id == "strong-old"


def test_search_applies_recency_ordering_for_last_queries(client_setup):
    """'who won the last final' must return 2025 first; the same
    search without recency intent keeps pure relevance order."""
    client, monkeypatch = client_setup

    import gateway.api as api_module

    class FakeRAG:
        def search(self, query, top_k=5, strategy=None, metadata_filter=None):
            return (
                [
                    result("ucl-2023", 2023, 9.0),
                    result("ucl-2025", 2025, 8.0),
                ],
                None,
            )

    monkeypatch.setattr(api_module, "rag_service", FakeRAG())

    response = client.post(
        "/v1/rag/search",
        headers={"X-API-Key": "redhat-demo-key"},
        json={"query": "who won the last champions league final"},
    )

    assert response.status_code == 200

    docs = [r["document_id"] for r in response.json()["results"]]

    assert docs == ["ucl-2025", "ucl-2023"]
