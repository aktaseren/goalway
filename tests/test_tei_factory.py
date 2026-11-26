from gateway.providers.tei import (
    create_tei_embedding_provider,
    load_tei_config,
)
from gateway.rag.embeddings import (
    OpenAICompatibleEmbeddingProvider,
)


def test_creates_tei_embedding_provider():
    config = load_tei_config()

    provider = create_tei_embedding_provider(
        config
    )

    assert isinstance(
        provider,
        OpenAICompatibleEmbeddingProvider,
    )
    assert provider.config.base_url == (
        "http://tei:8080/v1"
    )
    assert provider.config.extra["runtime"] == (
        "tei"
    )