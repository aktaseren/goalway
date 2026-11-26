from dataclasses import dataclass
from datetime import datetime

from gateway.rag.embeddings.base import EmbeddingProvider


@dataclass
class ProviderHealth:
    healthy: bool
    checked_at: datetime


class EmbeddingHealthManager:
    def __init__(self):
        self._providers: dict[str, EmbeddingProvider] = {}
        self._health: dict[str, ProviderHealth] = {}

    def register(
        self,
        name: str,
        provider: EmbeddingProvider,
    ):
        self._providers[name] = provider

    def refresh(self):
        now = datetime.utcnow()

        for name, provider in self._providers.items():

            self._health[name] = ProviderHealth(
                healthy=provider.health(),
                checked_at=now,
            )

    def is_healthy(
        self,
        name: str,
    ) -> bool:
        return self._health[name].healthy

    def healthy_providers(self):
        return [
            name
            for name, health in self._health.items()
            if health.healthy
        ]