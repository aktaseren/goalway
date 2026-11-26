from gateway.providers.common import ProviderConfig
from gateway.providers.generation import (
    OpenAICompatibleGenerationProvider,
    create_generation_provider,
    registered_generation_providers,
)


def test_factory_creates_openai_compatible_provider():
    provider = create_generation_provider(
        ProviderConfig(
            provider="openshift_ai",
            model="granite",
            base_url="http://model:8000/v1",
        )
    )

    assert isinstance(
        provider,
        OpenAICompatibleGenerationProvider,
    )


def test_generation_provider_aliases_are_registered():
    assert registered_generation_providers() == [
        "openai",
        "openai_compatible",
        "openshift_ai",
        "vllm",
    ]