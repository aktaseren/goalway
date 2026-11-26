from gateway.rag.embeddings.base import EmbeddingProvider

from gateway.rag.embeddings.config import EmbeddingConfig

from gateway.rag.embeddings.capabilities import (
    EmbeddingCapabilities,
)

# Populated lazily: sentence-transformers pulls in torch and is an
# optional dependency. Only fail when this provider is used.
SentenceTransformer = None


def _load_sentence_transformer():
    global SentenceTransformer

    if SentenceTransformer is None:
        try:
            from sentence_transformers import (
                SentenceTransformer as _STM,
            )
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceEmbeddingProvider requires the optional "
                "'sentence-transformers' package. "
                "Install it or use another embedding provider."
            ) from exc
        SentenceTransformer = _STM

    return SentenceTransformer

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(
            self,
            config: EmbeddingConfig,
    ):
        super().__init__(config)

        model_class = _load_sentence_transformer()

        self.model = model_class(
            self.config.model
            )

    def health(self):
        return self.model is not None

    def capabilities(self):
        return EmbeddingCapabilities(
            provider="huggingface",
            model=self.config.model,
            operations=frozenset({"embed"}),
            dimensions=self.config.dimensions,
            supports_batching=True,
            supports_streaming=False,
            supports_local_inference=True,
            max_batch_size=self.config.batch_size,
        )

    def embed(
            self,
            texts: list[str],
    ) -> list[list[float]]:
        
        return self.model.encode(
            texts,
            convert_to_numpy=False,
        )