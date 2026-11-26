from datetime import datetime

from gateway.rag.embeddings.base import EmbeddingProvider
from gateway.rag.embeddings.capabilities import EmbeddingCapabilities
from gateway.rag.embeddings.health_manager import (
    EmbeddingHealthManager,
)


class HealthyProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def health(self) -> bool:
        return True

    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider="healthy",
            model="test-model",
            dimensions=1,
            supports_batching=True,
            supports_streaming=False,
            supports_local_inference=True,
            max_batch_size=32,
        )


class UnhealthyProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def health(self) -> bool:
        return False

    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider="unhealthy",
            model="test-model",
            dimensions=1,
            supports_batching=True,
            supports_streaming=False,
            supports_local_inference=True,
            max_batch_size=32,
        )


def test_refresh_updates_provider_health():
    manager = EmbeddingHealthManager()

    manager.register(
        "healthy",
        HealthyProvider(),
    )

    manager.register(
        "unhealthy",
        UnhealthyProvider(),
    )

    manager.refresh()

    assert manager.is_healthy("healthy")
    assert not manager.is_healthy("unhealthy")


def test_healthy_providers_returns_only_healthy():
    manager = EmbeddingHealthManager()

    manager.register(
        "healthy",
        HealthyProvider(),
    )

    manager.register(
        "unhealthy",
        UnhealthyProvider(),
    )

    manager.refresh()

    assert manager.healthy_providers() == [
        "healthy",
    ]


def test_refresh_sets_timestamp():
    manager = EmbeddingHealthManager()

    manager.register(
        "healthy",
        HealthyProvider(),
    )

    manager.refresh()

    health = manager._health["healthy"]

    assert isinstance(
        health.checked_at,
        datetime,
    )


def test_multiple_refreshes_update_timestamp():
    manager = EmbeddingHealthManager()

    manager.register(
        "healthy",
        HealthyProvider(),
    )

    manager.refresh()
    first = manager._health["healthy"].checked_at

    manager.refresh()
    second = manager._health["healthy"].checked_at

    assert second >= first