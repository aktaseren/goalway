import os
from dataclasses import dataclass

from gateway.providers.common import ProviderConfig


@dataclass(frozen=True)
class EmbeddingConfig(ProviderConfig):
    dimensions: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError(
                "Embedding dimensions must be greater than zero"
            )

def load_embedding_config() -> EmbeddingConfig:
    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        "dummy",
    ).strip().lower()

    defaults = {
        "dummy": {
            "model": "dummy",
            "base_url": None,
        },
        "ollama": {
            "model": "embeddinggemma",
            "base_url": "http://ollama:11434",
        },
        "openai": {
            "model": "text-embedding-3-small",
            "base_url": "https://api.openai.com/v1",
        },
        "openai_compatible": {
            "model": "embedding-model",
            "base_url": "http://embedding-service:8000/v1",
        },
        "vllm": {
            "model": "embedding-model",
            "base_url": "http://vllm:8000/v1",
        },
        "huggingface": {
            "model": "BAAI/bge-small-en-v1.5",
            "base_url": None,
        },
    }

    provider_defaults = defaults.get(
        provider,
        {
            "model": "embedding-model",
            "base_url": None,
        },
    )

    dimensions_value = os.getenv(
        "EMBEDDING_DIMENSIONS"
    )
    base_url_value = (
        os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or provider_defaults["base_url"]
    )
    model_value = (
        os.getenv("EMBEDDING_MODEL")
        or os.getenv("OLLAMA_EMBEDDING_MODEL")
        or provider_defaults["model"]
    )
    timeout_value = (
        os.getenv("EMBEDDING_TIMEOUT_SECONDS")
        or os.getenv("OLLAMA_TIMEOUT_SECONDS")
        or "120"
    )

    return EmbeddingConfig(
        provider=provider,
        model=model_value,
        base_url=base_url_value,
        api_key=os.getenv("EMBEDDING_API_KEY"),
        timeout_seconds=int(timeout_value),
        dimensions=(
            int(dimensions_value)
            if dimensions_value
            else None
        ),
        batch_size=int(
            os.getenv(
                "EMBEDDING_BATCH_SIZE",
                "32",
            )
        ),
    )