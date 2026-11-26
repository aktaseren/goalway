from collections.abc import Callable

from gateway.providers.common import ProviderConfig
from gateway.providers.generation.base import GenerationProvider


GenerationProviderFactory = Callable[
    [ProviderConfig],
    GenerationProvider,
]


class GenerationProviderRegistry:
    def __init__(self):
        self._providers: dict[
            str,
            GenerationProviderFactory,
        ] = {}

    def register(
        self,
        name: str,
        factory: GenerationProviderFactory,
    ) -> None:
        key = name.lower()

        if key in self._providers:
            raise ValueError(
                f"Generation provider '{name}' is already registered"
            )

        self._providers[key] = factory

    def create(
        self,
        name: str,
        config: ProviderConfig,
    ) -> GenerationProvider:
        key = name.lower()

        if key not in self._providers:
            raise ValueError(
                f"Unsupported generation provider: {name}"
            )

        return self._providers[key](config)

    def registered_names(self) -> list[str]:
        return sorted(self._providers)