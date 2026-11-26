from gateway.rag.embeddings.config import (
    EmbeddingConfig,
    load_embedding_config,
)
from gateway.rag.embeddings.dummy import DummyEmbeddingProvider
from gateway.rag.embeddings.huggingface import (
    HuggingFaceEmbeddingProvider,
)
from gateway.rag.embeddings.ollama import OllamaEmbeddingProvider
from gateway.rag.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from gateway.rag.embeddings.registry import (
    EmbeddingProviderRegistry,
)

_registry = EmbeddingProviderRegistry()

_registry.register(
    "dummy",
    DummyEmbeddingProvider,
)

_registry.register(
    "ollama",
    OllamaEmbeddingProvider,
)

_registry.register(
    "openai",
    OpenAICompatibleEmbeddingProvider,
)

_registry.register(
    "openai_compatible",
    OpenAICompatibleEmbeddingProvider,
)

_registry.register(
    "vllm",
    OpenAICompatibleEmbeddingProvider,
)

_registry.register(
    "huggingface",
    HuggingFaceEmbeddingProvider,
)


def create_embedding_provider(
        config: EmbeddingConfig | None = None,
):
    resolved_config = config or load_embedding_config()

    return _registry.create(
        resolved_config.provider,
        resolved_config,
    )

def registered_embedding_providers() -> list[str]:
    return _registry.registered_names()