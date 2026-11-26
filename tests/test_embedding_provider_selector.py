import pytest

from gateway.rag.embeddings.health_manager import (
    EmbeddingHealthManager,
    ProviderHealth,
)
from gateway.rag.embeddings.provider_selector import (
    EmbeddingProviderSelector,
)


def test_selects_only_healthy_provider():
    manager = EmbeddingHealthManager()
    manager._health = {
        "ollama": ProviderHealth(
            healthy=True,
            checked_at=None,
        ),
        "openai": ProviderHealth(
            healthy=False,
            checked_at=None,
        ),
    }

    selector = EmbeddingProviderSelector(manager)

    assert selector.select() == "ollama"


def test_selects_first_healthy_provider_deterministically():
    manager = EmbeddingHealthManager()
    manager._health = {
        "ollama": ProviderHealth(
            healthy=True,
            checked_at=None,
        ),
        "huggingface": ProviderHealth(
            healthy=True,
            checked_at=None,
        ),
    }

    selector = EmbeddingProviderSelector(manager)

    assert selector.select() == "ollama"


def test_ignores_unhealthy_providers():
    manager = EmbeddingHealthManager()
    manager._health = {
        "ollama": ProviderHealth(
            healthy=False,
            checked_at=None,
        ),
        "huggingface": ProviderHealth(
            healthy=True,
            checked_at=None,
        ),
        "openai": ProviderHealth(
            healthy=False,
            checked_at=None,
        ),
    }

    selector = EmbeddingProviderSelector(manager)

    assert selector.select() == "huggingface"


def test_raises_when_no_healthy_provider_exists():
    manager = EmbeddingHealthManager()
    manager._health = {
        "ollama": ProviderHealth(
            healthy=False,
            checked_at=None,
        ),
        "openai": ProviderHealth(
            healthy=False,
            checked_at=None,
        ),
    }

    selector = EmbeddingProviderSelector(manager)

    with pytest.raises(
        RuntimeError,
        match="No healthy embedding providers available",
    ):
        selector.select()


def test_raises_when_health_state_is_empty():
    manager = EmbeddingHealthManager()
    selector = EmbeddingProviderSelector(manager)

    with pytest.raises(
        RuntimeError,
        match="No healthy embedding providers available",
    ):
        selector.select()