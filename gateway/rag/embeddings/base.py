from abc import ABC, abstractmethod

from gateway.providers.common import (
    Provider,
    ProviderCapabilities,
    ProviderConfig,
)

class EmbeddingProvider(ABC):
    def __init__(self, config: ProviderConfig | None = None):
        self.config = config

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
    
    @abstractmethod
    def health(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def capabilities(self,) -> ProviderCapabilities:
        raise NotImplementedError