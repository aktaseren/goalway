import json
import requests

MODEL_REGISTRY = {}
MODEL_HEALTH = {}

def load_model_registry(registry_file: str):
    global MODEL_REGISTRY

    with open(registry_file, "r") as f:
        MODEL_REGISTRY = json.load(f)["models"]

    return MODEL_REGISTRY

def get_model_config(model_name: str):
    return MODEL_REGISTRY.get(model_name)

def list_models():
    return list(MODEL_REGISTRY.keys())

def check_backend_health(model_name: str, model_config: dict):
    # Local providers (e.g. the built-in "rag" model) have no backend
    # service to probe -- they are always healthy.
    if not model_config.get("health_url"):
        MODEL_HEALTH[model_name] = True
        return

    try:
        response = requests.get(
            model_config["health_url"],
            timeout=5
        )
        MODEL_HEALTH[model_name] = response.status_code == 200
    except Exception:
        MODEL_HEALTH[model_name] = False

def refresh_health():
    for name, model_config in MODEL_REGISTRY.items():
        check_backend_health(name, model_config)
    return MODEL_HEALTH

def is_model_healthy(model_name: str):
    return MODEL_HEALTH.get(model_name, False)


def start_background_health_refresh(interval_seconds: int = 60):
    """
    probe all backend health endpoints on startup and keep
    refreshing them in the background so the gateway is usable
    immediately after a rollout (no manual /admin/reload needed).
    """
    import threading

    def _loop():
        while True:
            try:
                refresh_health()
            except Exception:
                pass
            import time

            time.sleep(interval_seconds)

    thread = threading.Thread(
        target=_loop,
        name="backend-health-refresher",
        daemon=True,
    )
    thread.start()
    return thread