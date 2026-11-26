"""Tests for gateway/rag/feedback.py (feedback loop)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from gateway.rag.feedback import FeedbackStore


def test_record_and_stats_in_memory(clean_feedback):
    store = FeedbackStore(db_path="")

    store.record(rating=1, query="pods", document_id="doc-1")
    store.record(rating=-1, query="routes", document_id="doc-2")
    store.record(rating=1, query="pods", document_id="doc-1")

    stats = store.stats()

    assert stats["total_feedback"] == 3
    assert stats["positive"] == 2
    assert stats["negative"] == 1
    assert stats["satisfaction_rate"] == round(2 / 3, 4)


def test_invalid_rating_raises(clean_feedback):
    store = FeedbackStore(db_path="")

    with pytest.raises(ValueError):
        store.record(rating=0)

    with pytest.raises(ValueError):
        store.record(rating=5)


def test_record_returns_entry(clean_feedback):
    store = FeedbackStore(db_path="")

    entry = store.record(
        rating=-1,
        query="pods",
        document_id="doc-1",
        comment="not helpful",
        username="alice",
        search_id="search-1",
    )

    assert entry["rating"] == -1
    assert entry["query"] == "pods"
    assert entry["document_id"] == "doc-1"
    assert entry["comment"] == "not helpful"
    assert entry["username"] == "alice"
    assert entry["search_id"] == "search-1"
    assert entry["feedback_id"].startswith("fb-")


def test_per_document_aggregation(clean_feedback):
    store = FeedbackStore(db_path="")

    store.record(rating=1, document_id="doc-1")
    store.record(rating=1, document_id="doc-1")
    store.record(rating=-1, document_id="doc-2")

    stats = store.stats()

    assert stats["per_document"]["doc-1"] == {"positive": 2, "negative": 0}
    assert stats["per_document"]["doc-2"] == {"positive": 0, "negative": 1}


def test_per_query_aggregation(clean_feedback):
    store = FeedbackStore(db_path="")

    store.record(rating=1, query="how do pods work")
    store.record(rating=-1, query="how do pods work")

    stats = store.stats()

    assert stats["per_query"]["how do pods work"] == {
        "positive": 1,
        "negative": 1,
    }


def test_stats_empty(clean_feedback):
    stats = FeedbackStore(db_path="").stats()

    assert stats["total_feedback"] == 0
    assert stats["satisfaction_rate"] is None
    assert stats["per_document"] == {}


def test_recent_comments(clean_feedback):
    store = FeedbackStore(db_path="")

    store.record(rating=1, query="q1", comment="great")
    store.record(rating=-1, query="q2")  # no comment -> excluded

    stats = store.stats()

    comments = stats["recent_comments"]
    assert len(comments) == 1
    assert comments[0]["query"] == "q1"
    assert comments[0]["comment"] == "great"
    assert comments[0]["rating"] == 1


def test_sqlite_persistence(tmp_path):
    db_path = str(tmp_path / "feedback.db")

    store = FeedbackStore(db_path=db_path)
    store.record(rating=1, query="pods", document_id="doc-1")
    store.record(rating=-1, query="routes", document_id="doc-2")

    # A brand new store instance on the same file sees the data.
    reopened = FeedbackStore(db_path=db_path)
    stats = reopened.stats()

    assert stats["total_feedback"] == 2
    assert stats["positive"] == 1
    assert stats["negative"] == 1
    assert stats["satisfaction_rate"] == 0.5
    assert len(reopened.all()) == 2


def test_in_memory_store_isolated_between_instances(clean_feedback):
    FeedbackStore(db_path="").record(rating=1, query="q")

    # New instance on the same in-memory backend shares the global
    # store; the clean_feedback fixture guarantees isolation between
    # *tests*.
    stats = FeedbackStore(db_path="").stats()

    assert stats["total_feedback"] == 1
