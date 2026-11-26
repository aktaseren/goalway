from gateway.rag.embeddings.base import EmbeddingProvider

from gateway.rag.embeddings.capabilities import EmbeddingCapabilities

from gateway.rag.embeddings.config import EmbeddingConfig


class DummyEmbeddingProvider(EmbeddingProvider):
    def __init__(
            self,
            config: EmbeddingConfig | None = None,
    ):
        super().__init__(
            config
            or EmbeddingConfig(
                provider="dummy",
                model="dummy",
            )
        )

    def health(self) -> bool:
        return True

    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider="dummy",
            model="dummy",
            operations=frozenset({"embed"}),
            dimensions=3,
            supports_batching=True,
            supports_streaming=False,
            supports_local_inference=True,
            max_batch_size=None,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float(len(text)),
                float(text.count(" ")),
                float(text.count(".")),
            ]
            for text in texts
        ]