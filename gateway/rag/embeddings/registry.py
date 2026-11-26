import inspect
from collections.abc import Callable

from gateway.rag.embeddings.base import EmbeddingProvider
from gateway.rag.embeddings.config import EmbeddingConfig


class EmbeddingProviderRegistry:
    def __init__(self):
        self.providers: dict[
            str,
            Callable[..., EmbeddingProvider],
        ] = {}

    def register(
            self,
            name: str,
            factory: Callable[..., EmbeddingProvider],
    ) -> None:
        key = name.lower()

        if key in self.providers:
            raise ValueError(
                f"Embedding provider '{name}' is already registered."
            )

        self.providers[key] = factory

    def create(
            self,
            name: str,
            config: EmbeddingConfig | None = None,
    ) -> EmbeddingProvider:
        key = name.lower()

        if key not in self.providers:
            raise ValueError(
                f"Unsupported embedding provider: {name}"
            )

        factory = self.providers[key]

        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return factory(config)

        parameters = list(signature.parameters.values())
        if not parameters:
            return factory()

        if any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        ):
            return factory(config)

        if config is None:
            return factory()

        return factory(config)

    def registered_names(self) -> list[str]:
        return sorted(self.providers.keys())