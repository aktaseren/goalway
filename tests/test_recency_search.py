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

    data = response.json()
    docs = [r["document_id"] for r in data["results"]]

    assert docs == ["ucl-2025", "ucl-2023"]

    # the horizon is disclosed so a stale "last" is visible as such
    assert data["knowledge_horizon"] == 2025


def test_search_without_recency_intent_has_no_horizon(client_setup):
    client, monkeypatch = client_setup

    import gateway.api as api_module

    class FakeRAG:
        def search(self, query, top_k=5, strategy=None, metadata_filter=None):
            return ([result("ucl-2023", 2023, 9.0)], None)

    monkeypatch.setattr(api_module, "rag_service", FakeRAG())

    response = client.post(
        "/v1/rag/search",
        headers={"X-API-Key": "redhat-demo-key"},
        json={"query": "who won the 2023 champions league final"},
    )

    assert response.status_code == 200
    assert response.json()["knowledge_horizon"] is None


def test_missing_recency_years():
    from datetime import date

    from gateway.rag.qa import missing_recency_years, recency_warning

    today = date.today()
    expected = today.year if today.month >= 6 else today.year - 1

    # KB current through the expected year -> no gap
    assert missing_recency_years(expected) == []
    assert recency_warning(expected) is None

    # one year stale -> exactly one missing year, warning names it
    assert missing_recency_years(expected - 1) == [expected]
    assert recency_warning(expected - 1) == (
        f"No info found for {expected} — the latest information in "
        f"the knowledge base is from {expected - 1}"
    )

    # multiple stale years -> range form
    missing = missing_recency_years(expected - 3)
    assert len(missing) == 3
    assert recency_warning(expected - 3).startswith(
        f"No info found for {expected - 2}-{expected}"
    )

    assert missing_recency_years(None) == []


def test_search_warns_when_years_are_missing(client_setup):
    """Stale KB + 'last final' query -> 'No info found for <year>'
    warning in the search response."""
    from datetime import date

    client, monkeypatch = client_setup

    import gateway.api as api_module

    today = date.today()
    expected = today.year if today.month >= 6 else today.year - 1

    class FakeRAG:
        def search(self, query, top_k=5, strategy=None, metadata_filter=None):
            # knowledge base is one year stale
            return ([result("ucl-stale", expected - 1, 9.0)], None)

    monkeypatch.setattr(api_module, "rag_service", FakeRAG())

    response = client.post(
        "/v1/rag/search",
        headers={"X-API-Key": "redhat-demo-key"},
        json={"query": "who won the last champions league final"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["knowledge_horizon"] == expected - 1
    assert data["recency_warning"] == (
        f"No info found for {expected} — the latest information in "
        f"the knowledge base is from {expected - 1}"
    )


def test_chat_prepends_recency_warning(monkeypatch):
    """Stale KB + 'last champion' question -> the chat answer leads
    with the 'No info found for <year>' warning."""
    from datetime import date

    import gateway.api as api_module
    from gateway.rag.models import DocumentChunk, SearchResult

    UCL = (
        "Real Madrid won the 2024 Champions League final, beating "
        "Borussia Dortmund 2-0 at Wembley."
    )

    class StaleRAG:
        def search(self, query, top_k=5, **kwargs):
            today = date.today()
            expected = today.year if today.month >= 6 else today.year - 1

            return (
                [
                    SearchResult(
                        chunk=DocumentChunk(
                            chunk_id="ucl-stale-0",
                            document_id="ucl-stale",
                            text=UCL,
                            position=0,
                            metadata={"year": expected - 1},
                        ),
                        score=0.9,
                    )
                ],
                None,
            )

        def get_document(self, document_id):
            return []

    monkeypatch.setattr(api_module, "rag_service", StaleRAG())

    from fastapi.testclient import TestClient

    from gateway.main import app

    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "rag",
            "messages": [
                {
                    "role": "user",
                    "content": "who won the last champions league final",
                }
            ],
        },
    )

    assert response.status_code == 200

    content = response.json()["choices"][0]["message"]["content"]

    assert content.startswith("⚠ No info found for ")
    assert "Real Madrid" in content  # the stale answer still follows
