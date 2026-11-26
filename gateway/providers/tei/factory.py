from gateway.providers.tei.client import TEIClient
from gateway.providers.tei.config import (
    TEIConfig,
    load_tei_config,
)
from gateway.rag.embeddings import (
    EmbeddingProvider,
    create_embedding_provider,
)


def create_tei_embedding_provider(
    config: TEIConfig | None = None,
) -> EmbeddingProvider:
    resolved = config or load_tei_config()

    return create_embedding_provider(
        resolved.embedding
    )


def create_tei_client(
    config: TEIConfig | None = None,
) -> TEIClient:
    return TEIClient(
        config or load_tei_config()
    )