PROVIDERS = {}

def register_auth_provider(name, provider):
    PROVIDERS[name] = provider

def get_auth_provider(name):
    return PROVIDERS.get(name)