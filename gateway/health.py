from gateway.registry import list_models, MODEL_HEALTH

def health_response():
    return {
        "status": "healthy",
        "models": list_models(),
        "model_health": MODEL_HEALTH,
    }

