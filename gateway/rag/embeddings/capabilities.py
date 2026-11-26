from dataclasses import dataclass

from gateway.providers.common import ProviderCapabilities


@dataclass(frozen=True)
class EmbeddingCapabilities(ProviderCapabilities):
    dimensions: int | None = None
