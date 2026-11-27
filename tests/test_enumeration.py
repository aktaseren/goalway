"""Tests for enumeration questions ("top 5 news")."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import gateway.api as api_module
import gateway.ratelimit as ratelimit_module
from gateway.main import app
from gateway.rag.models import DocumentChunk, SearchResult
from gateway.rag.qa import (
    extract_requested_count,
    gate_coverage,
)

from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_rate_limit():
    ratelimit_module.REQUEST_HISTORY.clear()
    yield
    ratelimit_module.REQUEST_HISTORY.clear()


def test_count_extraction():
    assert extract_requested_count("tell me top 5 todays sport news") == 5
    assert extract_requested_count("latest 3 headlines") == 3
    assert extract_requested_count("show me the first 2 stories") == 2
    assert extract_requested_count("give me 7 news items") == 7

    # years and bare numbers are not counts
    assert extract_requested_count("who won the 2023 champions league final") is None
    assert extract_requested_count("who won the last champions league final") is None


def test_gate_coverage_none_for_pure_enumeration():
    # "top 5 news" has no topical terms after shape words -> None
    assert (
        gate_coverage(
            "tell me top 5 todays sport news",
            "Arsenal won the derby.",
            extra_stop={"top", "news", "sport", "sports", "today", "todays"},
        )
        is None
    )

    # "top 5 messi news" still gates on messi
    assert (
        gate_coverage(
            "top 5 messi news",
            "Arsenal won the derby.",
            extra_stop={"top", "news"},
        )
        == 0.0
    )


NEWS_ITEMS = [
    ("news-1", 2026, "Arsenal beat Chelsea 2-1 in the London derby."),
    ("news-2", 2026, "Liverpool sign a new striker on a free transfer."),
    ("news-3", 2026, "Real Madrid training update ahead of the clasico."),
    ("news-4", 2026, "Bayern Munich extend their coach's contract."),
    ("news-5", 2026, "Inter Milan reach the cup semi final."),
    ("news-6", 2026, "PSG rest their stars for the weekend fixture."),
]


class FakeNewsRAG:
    def search(self, query, top_k=5, **kwargs):
        return (
            [
                SearchResult(
                    chunk=DocumentChunk(
                        chunk_id=f"{doc_id}-0",
                        document_id=doc_id,
                        text=text,
                        position=0,
                        metadata={"year": year, "source": "test-feed"},
                    ),
                    score=0.9,
                )
                for doc_id, year, text in NEWS_ITEMS
            ],
            None,
        )

    def get_document(self, document_id):
        return []


def test_top_5_news_returns_five_numbered_items(monkeypatch):
    monkeypatch.setattr(api_module, "rag_service", FakeNewsRAG())

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "rag",
            "messages": [
                {
                    "role": "user",
                    "content": "tell me top 5 todays sport news",
                }
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # exactly five numbered items
    lines = [
        line
        for line in content.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    assert len(lines) == 5
    assert lines[0].startswith("1. ")
    assert lines[4].startswith("5. ")

    # five citations, one per item
    assert len(data["gateway"]["citations"]) == 5


def test_top_3_news_returns_three_items(monkeypatch):
    monkeypatch.setattr(api_module, "rag_service", FakeNewsRAG())

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "redhat-demo-key"},
        json={
            "model": "rag",
            "messages": [
                {"role": "user", "content": "top 3 news"}
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    lines = [
        line
        for line in content.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    assert len(lines) == 3
