from abc import abstractmethod
from collections.abc import Iterator

from gateway.providers.common import (
    Provider,
    ProviderCapabilities,
    ProviderConfig,
)
from gateway.providers.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


class GenerationProvider(Provider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        raise NotImplementedError

    def stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[str]:
        raise NotImplementedError(
            "Streaming is not supported by this provider"
        )

    @abstractmethod
    def health(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError