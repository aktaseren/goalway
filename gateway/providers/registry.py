PROVIDERS = {}

def register_provider(name, provider_class):
    PROVIDERS[name] = provider_class

def get_provider_class(name):
    return PROVIDERS.get(name)