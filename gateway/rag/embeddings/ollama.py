import requests

from gateway.rag.embeddings.base import EmbeddingProvider
from gateway.rag.embeddings.config import EmbeddingConfig

import requests

from gateway.rag.embeddings.capabilities import (
    EmbeddingCapabilities,
)

class OllamaEmbeddingProvider(EmbeddingProvider):
    def health(self) -> bool:
        try:
            response = requests.get(
                f"{self.config.base_url}/api/version",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider="ollama",
            model=self.config.model,
            operations=frozenset({"embed"}),
            dimensions=self.config.dimensions,
            supports_batching=True,
            supports_streaming=False,
            supports_local_inference=True,
            max_batch_size=self.config.batch_size,
        )

    def __init__(
            self,
            config: EmbeddingConfig | None = None,
            *,
            base_url: str | None = None,
            model: str | None = None,
            timeout_seconds: int | None = None,
    ):
        if isinstance(config, EmbeddingConfig):
            resolved_config = config
            resolved_base_url = config.base_url if config.base_url is not None else base_url
            resolved_model = config.model or model or "embeddinggemma"
            resolved_timeout = (
                config.timeout_seconds
                if config.timeout_seconds is not None
                else (timeout_seconds if timeout_seconds is not None else 120)
            )
        else:
            resolved_config = EmbeddingConfig(
                provider="ollama",
                model=model or "embeddinggemma",
                base_url=base_url,
                timeout_seconds=timeout_seconds if timeout_seconds is not None else 120,
            )
            resolved_base_url = base_url
            resolved_model = model or "embeddinggemma"
            resolved_timeout = timeout_seconds if timeout_seconds is not None else 120

        self.config = resolved_config
        self.base_url = (resolved_base_url or "").rstrip("/")
        self.model = resolved_model
        self.timeout_seconds = resolved_timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise ValueError("Ollama response did not contain an embeddings list")

        if len(embeddings) != len(texts):
            raise ValueError(
                "Ollama returned a different number of embeddings than inputs"
            )

        return embeddings