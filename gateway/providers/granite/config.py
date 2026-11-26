import os
from dataclasses import dataclass

from gateway.providers.common import ProviderConfig
from gateway.rag.embeddings.config import EmbeddingConfig


DEFAULT_GRANITE_GENERATION_MODEL = (
    "ibm-granite/granite-4.1-3b"
)

DEFAULT_GRANITE_EMBEDDING_MODEL = (
    "ibm-granite/granite-embedding-small-english-r2"
)


@dataclass(frozen=True)
class GraniteConfig:
    generation: ProviderConfig
    embedding: EmbeddingConfig
    embedding_backend: str


def load_granite_config() -> GraniteConfig:
    generation_backend = os.getenv(
        "GRANITE_GENERATION_BACKEND",
        "openshift_ai",
    ).strip().lower()

    embedding_backend = os.getenv(
        "GRANITE_EMBEDDING_BACKEND",
        "huggingface",
    ).strip().lower()

    generation = ProviderConfig(
        provider=generation_backend,
        model=os.getenv(
            "GRANITE_GENERATION_MODEL",
            DEFAULT_GRANITE_GENERATION_MODEL,
        ),
        base_url=os.getenv(
            "GRANITE_GENERATION_BASE_URL",
            "http://granite-service:8080/v1",
        ),
        api_key=os.getenv(
            "GRANITE_GENERATION_API_KEY"
        ),
        timeout_seconds=int(
            os.getenv(
                "GRANITE_GENERATION_TIMEOUT_SECONDS",
                "600",
            )
        ),
        batch_size=1,
        extra={
            "family": "granite",
            "operation": "generation",
        },
    )

    embedding_base_url = os.getenv(
        "GRANITE_EMBEDDING_BASE_URL"
    )

    if embedding_backend in {
        "vllm",
        "openai",
        "openai_compatible",
        "openshift_ai",
    }:
        embedding_base_url = (
            embedding_base_url
            or "http://granite-embedding-service:8080/v1"
        )

    embedding = EmbeddingConfig(
        provider=embedding_backend,
        model=os.getenv(
            "GRANITE_EMBEDDING_MODEL",
            DEFAULT_GRANITE_EMBEDDING_MODEL,
        ),
        base_url=embedding_base_url,
        api_key=os.getenv(
            "GRANITE_EMBEDDING_API_KEY"
        ),
        timeout_seconds=int(
            os.getenv(
                "GRANITE_EMBEDDING_TIMEOUT_SECONDS",
                "120",
            )
        ),
        dimensions=int(
            os.getenv(
                "GRANITE_EMBEDDING_DIMENSIONS",
                "384",
            )
        ),
        batch_size=int(
            os.getenv(
                "GRANITE_EMBEDDING_BATCH_SIZE",
                "32",
            )
        ),
        extra={
            "family": "granite",
            "operation": "embedding",
        },
    )

    return GraniteConfig(
        generation=generation,
        embedding=embedding,
        embedding_backend=embedding_backend,
    )