from gateway.providers.generation import (
    GenerationProvider,
    create_generation_provider,
)
from gateway.providers.granite.config import (
    GraniteConfig,
    load_granite_config,
)
from gateway.rag.embeddings import (
    EmbeddingProvider,
    create_embedding_provider,
)


def create_granite_generation_provider(
    config: GraniteConfig | None = None,
) -> GenerationProvider:
    resolved = config or load_granite_config()

    return create_generation_provider(
        resolved.generation
    )


def create_granite_embedding_provider(
    config: GraniteConfig | None = None,
) -> EmbeddingProvider:
    resolved = config or load_granite_config()

    return create_embedding_provider(
        resolved.embedding
    )