from gateway.registry import list_models, get_model_config, is_model_healthy

def get_failover_model(model: str):
    """
    Return another healthy model if the requested model is unavailable.
    """

    for name in list_models():
        config = get_model_config(name)
        if (
            name != model
            and config.get("enabled", False)
            and is_model_healthy(name)
        ):
            return name
    return None