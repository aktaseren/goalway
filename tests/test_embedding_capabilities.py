from gateway.rag.embeddings.config import EmbeddingConfig
from gateway.rag.embeddings.dummy import DummyEmbeddingProvider


def test_dummy_capabilities():

    provider = DummyEmbeddingProvider()

    caps = provider.capabilities()

    assert caps.provider == "dummy"
    assert caps.supports_batching
    assert caps.supports_local_inference