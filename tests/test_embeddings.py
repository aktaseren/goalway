import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.rag.embeddings import DummyEmbeddingProvider

def test_dummy_embedding_provider_returns_one_vector_per_text():
    provider = DummyEmbeddingProvider()

    embeddings = provider.embed(
        [
            "Hello Red Hat AI Gateway",
            "OpenShift AI RAG",
        ]
    )

    assert len(embeddings) == 2
    assert all(len(vector) == 3 for vector in embeddings)