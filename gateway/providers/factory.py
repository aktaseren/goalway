from gateway.providers.registry import get_provider_class

# Import providers so they register themselves
import gateway.providers.openai_compatible
import gateway.providers.ollama

def get_provider(provider_name):
    provider_class = get_provider_class(provider_name)

    if provider_class is None:
        raise ValueError(f"Unknown Provider: {provider_name}")
    return provider_class()