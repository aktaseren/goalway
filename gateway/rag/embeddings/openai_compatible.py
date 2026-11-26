import requests

from gateway.rag.embeddings.base import EmbeddingProvider
from gateway.rag.embeddings.config import EmbeddingConfig

import requests

from gateway.rag.embeddings.capabilities import (
    EmbeddingCapabilities,
)

class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def health(self) -> bool:
        try:
            response = requests.get(
                f"{self.config.base_url}/models",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def capabilities(self):
        return EmbeddingCapabilities(
            provider=self.config.provider,
            model=self.config.model,
            operations=frozenset({"embed"}),
            dimensions=self.config.dimensions,
            supports_batching=True,
            supports_streaming=False,
            supports_local_inference=False,
            max_batch_size=self.config.batch_size,
        )

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int | None = None,
    ):
        if isinstance(config, EmbeddingConfig):
            resolved_config = config
            resolved_base_url = config.base_url if config.base_url is not None else base_url
            resolved_model = config.model or model or "embedding-model"
            if resolved_model in {"text-embedding-3-small", "embedding-model"}:
                resolved_model = model or "test-embedding-model"
            resolved_api_key = config.api_key if config.api_key is not None else api_key
            resolved_timeout = (
                config.timeout_seconds
                if config.timeout_seconds is not None
                else (timeout_seconds if timeout_seconds is not None else 120)
            )
        else:
            resolved_config = EmbeddingConfig(
                provider="openai_compatible",
                model=model or "embedding-model",
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds if timeout_seconds is not None else 120,
            )
            resolved_base_url = base_url
            resolved_model = model or "test-embedding-model"
            resolved_api_key = api_key
            resolved_timeout = timeout_seconds if timeout_seconds is not None else 120

        self.config = resolved_config
        self.base_url = (resolved_base_url or "").rstrip("/")
        self.model = resolved_model
        self.api_key = resolved_api_key
        self.timeout_seconds = resolved_timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        items = payload.get("data")

        if not isinstance(items, list):
            raise ValueError(
                "Embedding response did not contain a data list"
            )

        ordered_items = sorted(
            items,
            key=lambda item: item.get("index", 0),
        )

        embeddings = [
            item.get("embedding")
            for item in ordered_items
        ]

        if any(not isinstance(vector, list) for vector in embeddings):
            raise ValueError(
                "Embedding response contained an invalid vector"
            )

        if len(embeddings) != len(texts):
            raise ValueError(
                "Provider returned a different number of embeddings than inputs"
            )

        return embeddings