"""
RAG feedback loop .

Users rate search results and answers (thumbs up/down + optional
comment). Feedback is stored with an optional SQLite backend and
aggregated per document and per query so retrieval quality can be
tracked over time -- the raw material for tuning chunk sizes,
strategies and rerankers.
"""

import json
import logging
import sqlite3
import threading
import time
import uuid

from gateway.config import FEEDBACK_DB_PATH

logger = logging.getLogger("goalway.feedback")

_lock = threading.Lock()
_memory_store: list[dict] = []


def _connect(path: str):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id TEXT NOT NULL,
            search_id TEXT,
            query TEXT,
            document_id TEXT,
            rating INTEGER NOT NULL,
            comment TEXT,
            username TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    return connection


class FeedbackStore:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path or FEEDBACK_DB_PATH
        self._connection = None

        if self.db_path:
            self._connection = _connect(self.db_path)
            logger.info("feedback store: sqlite at %s", self.db_path)
        else:
            logger.info("feedback store: in-memory")

    def record(
        self,
        rating: int,
        query: str = "",
        document_id: str = "",
        comment: str = "",
        username: str = "",
        search_id: str = "",
    ) -> dict:
        """Store one feedback event. rating: 1 (positive) or -1 (negative)."""
        if rating not in (1, -1):
            raise ValueError("rating must be 1 or -1")

        entry = {
            # timestamp + random suffix: unique even under concurrency
            "feedback_id": f"fb-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}",
            "search_id": search_id,
            "query": query,
            "document_id": document_id,
            "rating": rating,
            "comment": comment,
            "username": username,
            "created_at": time.time(),
        }

        with _lock:
            if self._connection is not None:
                self._connection.execute(
                    """
                    INSERT INTO feedback
                    (feedback_id, search_id, query, document_id,
                     rating, comment, username, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry["feedback_id"],
                        entry["search_id"],
                        entry["query"],
                        entry["document_id"],
                        entry["rating"],
                        entry["comment"],
                        entry["username"],
                        entry["created_at"],
                    ),
                )
                self._connection.commit()
            else:
                _memory_store.append(entry)

        return entry

    def stats(self) -> dict:
        """Aggregate feedback: overall and per document/query."""
        entries = self.all()

        total = len(entries)
        positive = sum(1 for e in entries if e["rating"] == 1)
        negative = total - positive

        per_document: dict[str, dict] = {}
        per_query: dict[str, dict] = {}

        for entry in entries:
            for grouping, key in (
                (per_document, entry["document_id"]),
                (per_query, entry["query"]),
            ):
                if not key:
                    continue

                if key not in grouping:
                    grouping[key] = {"positive": 0, "negative": 0}

                if entry["rating"] == 1:
                    grouping[key]["positive"] += 1
                else:
                    grouping[key]["negative"] += 1

        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": round(positive / total, 4) if total else None,
            "per_document": per_document,
            "per_query": per_query,
            "recent_comments": [
                {
                    "query": e["query"],
                    "comment": e["comment"],
                    "rating": e["rating"],
                }
                for e in entries[-10:]
                if e["comment"]
            ],
        }

    def all(self) -> list[dict]:
        with _lock:
            if self._connection is not None:
                rows = self._connection.execute(
                    """
                    SELECT feedback_id, search_id, query, document_id,
                           rating, comment, username, created_at
                    FROM feedback ORDER BY created_at
                    """
                ).fetchall()

                return [
                    {
                        "feedback_id": row[0],
                        "search_id": row[1],
                        "query": row[2],
                        "document_id": row[3],
                        "rating": row[4],
                        "comment": row[5],
                        "username": row[6],
                        "created_at": row[7],
                    }
                    for row in rows
                ]

            return list(_memory_store)


_default_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    global _default_store

    if _default_store is None:
        _default_store = FeedbackStore()

    return _default_store
