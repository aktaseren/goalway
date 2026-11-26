import os

from gateway.providers.common import ProviderConfig
from gateway.providers.generation.openai_compatible import (
    OpenAICompatibleGenerationProvider,
)
from gateway.providers.generation.registry import (
    GenerationProviderRegistry,
)


_registry = GenerationProviderRegistry()

for provider_name in (
    "openai",
    "openai_compatible",
    "vllm",
    "openshift_ai",
):
    _registry.register(
        provider_name,
        OpenAICompatibleGenerationProvider,
    )


def load_generation_config() -> ProviderConfig:
    provider = os.getenv(
        "GENERATION_PROVIDER",
        "openai_compatible",
    ).strip().lower()

    return ProviderConfig(
        provider=provider,
        model=os.getenv(
            "GENERATION_MODEL",
            "granite",
        ),
        base_url=os.getenv(
            "GENERATION_BASE_URL",
            "http://granite-service:8080/v1",
        ),
        api_key=os.getenv(
            "GENERATION_API_KEY"
        ),
        timeout_seconds=int(
            os.getenv(
                "GENERATION_TIMEOUT_SECONDS",
                "600",
            )
        ),
        batch_size=1,
    )


def create_generation_provider(
    config: ProviderConfig | None = None,
):
    resolved_config = config or load_generation_config()

    return _registry.create(
        resolved_config.provider,
        resolved_config,
    )


def registered_generation_providers() -> list[str]:
    return _registry.registered_names()