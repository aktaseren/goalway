from gateway.auth_providers.registry import get_auth_provider

# force provider registration
import gateway.auth_providers.api_key

def create_auth_provider(name):
    provider = get_auth_provider(name)

    if provider is None:
        raise ValueError(f"Unknown auth provider: {name}")
    return provider()