import pytest

from gateway.providers.common import (
    Provider,
    ProviderCapabilities,
    ProviderConfig,
)


class FakeProvider(Provider):
    def health(self) -> bool:
        return True

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.config.provider,
            model=self.config.model,
            operations=frozenset({"embed", "chat"}),
            supports_batching=True,
            supports_streaming=True,
            supports_local_inference=True,
            max_batch_size=16,
            metadata={"source": "test"},
        )


def test_provider_config_validates_required_fields():
    with pytest.raises(
        ValueError,
        match="Provider name cannot be empty",
    ):
        ProviderConfig(
            provider="",
            model="test-model",
        )

    with pytest.raises(
        ValueError,
        match="Provider model cannot be empty",
    ):
        ProviderConfig(
            provider="test",
            model="",
        )


def test_provider_config_validates_numeric_values():
    with pytest.raises(
        ValueError,
        match="timeout",
    ):
        ProviderConfig(
            provider="test",
            model="test-model",
            timeout_seconds=0,
        )

    with pytest.raises(
        ValueError,
        match="batch size",
    ):
        ProviderConfig(
            provider="test",
            model="test-model",
            batch_size=0,
        )


def test_provider_stores_configuration():
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        base_url="http://fake:8000",
        timeout_seconds=30,
        batch_size=16,
        extra={"dimensions": 768},
    )

    provider = FakeProvider(config)

    assert provider.config == config
    assert provider.health() is True


def test_capabilities_supports_operations():
    provider = FakeProvider(
        ProviderConfig(
            provider="fake",
            model="fake-model",
        )
    )

    capabilities = provider.capabilities()

    assert capabilities.supports("embed")
    assert capabilities.supports("chat")
    assert not capabilities.supports("rerank")
    assert capabilities.max_batch_size == 16
    assert capabilities.metadata["source"] == "test"