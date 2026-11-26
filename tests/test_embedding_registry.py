import pytest

from gateway.providers.common import ProviderCapabilities
from gateway.rag.embeddings.base import EmbeddingProvider
from gateway.rag.embeddings.registry import EmbeddingProviderRegistry

from gateway.rag.embeddings.config import EmbeddingConfig

from gateway.rag.embeddings.dummy import DummyEmbeddingProvider


class FakeProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def health(self) -> bool:
        return True

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider="fake",
            model="fake-model",
            operations=frozenset({"embed"}),
        )


def test_register_and_create_provider():
    registry = EmbeddingProviderRegistry()

    registry.register(
        "fake",
        FakeProvider,
    )

    provider = registry.create("fake")

    assert isinstance(provider, FakeProvider)


def test_duplicate_registration_is_rejected():
    registry = EmbeddingProviderRegistry()

    registry.register(
        "fake",
        FakeProvider,
    )

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(
            "fake",
            FakeProvider,
        )


def test_unknown_provider_is_rejected():
    registry = EmbeddingProviderRegistry()

    with pytest.raises(
        ValueError,
        match="Unsupported embedding provider",
    ):
        registry.create("missing")


def test_registered_names_are_sorted():
    registry = EmbeddingProviderRegistry()

    registry.register("zeta", FakeProvider)
    registry.register("alpha", FakeProvider)

    assert registry.registered_names() == [
        "alpha",
        "zeta",
    ]

def test_configuration_is_passed_to_provider():
    registry = EmbeddingProviderRegistry()

    registry.register(
        "dummy",
        DummyEmbeddingProvider,
    )

    config = EmbeddingConfig(
        provider = "dummy",
        model="enterprise-model",
    )

    provider = registry.create(
        "dummy",
        config,
    )

    assert provider.config.model == "enterprise-model"