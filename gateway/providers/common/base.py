from abc import ABC, abstractmethod

from gateway.providers.common.capabilities import (
    ProviderCapabilities,
)

from gateway.providers.common.config import ProviderConfig

class Provider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @abstractmethod
    def health(self) -> bool:
        """Return whether the provider is currently available."""
        raise NotImplementedError
    
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Describe the operations and features supported by the provider."""
        raise NotImplementedError