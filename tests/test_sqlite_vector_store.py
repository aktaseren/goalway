import tempfile

from gateway.rag.models import DocumentChunk
from gateway.rag.sqlite_vector_store import SQLiteVectorStore


def test_sqlite_store_persists_vectors():

    with tempfile.NamedTemporaryFile() as db:

        store = SQLiteVectorStore(db.name)

        chunk = DocumentChunk(
            chunk_id="chunk1",
            document_id="doc1",
            position=0,
            text="OpenShift AI",
            metadata={},
        )

        store.add(
            [chunk],
            [[1.0, 0.0, 0.0]],
        )

        del store

        reopened = SQLiteVectorStore(db.name)

        results = reopened.search(
            [1.0, 0.0, 0.0],
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].chunk.text == "OpenShift AI"