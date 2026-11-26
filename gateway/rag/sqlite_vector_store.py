import json
import math
import sqlite3
from pathlib import Path

from gateway.rag.models import DocumentChunk, SearchResult
from gateway.rag.vector_store import VectorStore


class SQLiteVectorStore(VectorStore):
    def __init__(self, db_path: str = "vectors.db"):
        self.db_path = db_path

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match "
                "the number of embeddings"
            )

        rows = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.position,
                chunk.text,
                json.dumps(chunk.metadata),
                json.dumps(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        document_ids = {chunk.document_id for chunk in chunks}

        with self._connect() as connection:
            # Replace-by-document: re-ingesting a document_id drops
            # its previous chunks first (a changed text may produce
            # different chunk ids, leaving stale rows behind).
            for document_id in document_ids:
                connection.execute(
                    "DELETE FROM vectors WHERE document_id = ?",
                    (document_id,),
                )

            connection.executemany(
                """
                INSERT INTO vectors (
                    chunk_id,
                    document_id,
                    position,
                    text,
                    metadata,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def all_chunks(self) -> list[DocumentChunk]:
        """Every stored chunk (used to rebuild the lexical index)."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, position, text, metadata
                FROM vectors
                """
            ).fetchall()

        return [
            DocumentChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                position=row["position"],
                text=row["text"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks of a document. Returns rows deleted."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM vectors WHERE document_id = ?",
                (document_id,),
            )

            return cursor.rowcount

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    position,
                    text,
                    metadata,
                    embedding
                FROM vectors
                """
            ).fetchall()

        results: list[SearchResult] = []

        for row in rows:
            stored_embedding = json.loads(row["embedding"])

            score = self._cosine_similarity(
                query_embedding,
                stored_embedding,
            )

            chunk = DocumentChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                position=row["position"],
                text=row["text"],
                metadata=json.loads(row["metadata"]),
            )

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM vectors"
            ).fetchone()

        return int(row["total"])

    @staticmethod
    def _cosine_similarity(
        first: list[float],
        second: list[float],
    ) -> float:
        if len(first) != len(second):
            raise ValueError("Embedding dimensions must match")

        dot_product = sum(
            left * right
            for left, right in zip(first, second)
        )

        first_norm = math.sqrt(
            sum(value * value for value in first)
        )
        second_norm = math.sqrt(
            sum(value * value for value in second)
        )

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return dot_product / (first_norm * second_norm)