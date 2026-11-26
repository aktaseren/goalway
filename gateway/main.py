from fastapi import FastAPI
from fastapi.responses import FileResponse

import os

from gateway.api import router, rag_router
from gateway.config import MODEL_REGISTRY_FILE
from gateway.keys import load_api_keys
from gateway.registry import load_model_registry
from gateway.telemetry import setup_tracing
from gateway.config import MCP_ENABLED

if MCP_ENABLED:
    from gateway.mcp import mcp_router

load_model_registry(MODEL_REGISTRY_FILE)

# API key store (roles + quotas), hot-reloadable
load_api_keys()

setup_tracing()

app = FastAPI(title="Goalway")

app.include_router(router)
app.include_router(rag_router)

if MCP_ENABLED:
    app.include_router(mcp_router)


@app.on_event("startup")
def startup_tasks():
    # probe backend health immediately so the gateway is usable
    # right after rollout (no manual /admin/reload needed).
    from gateway.registry import start_background_health_refresh

    start_background_health_refresh()


# built-in demo console (static/index.html)
@app.get("/ui", include_in_schema=False)
def demo_ui():
    return FileResponse(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "static",
            "index.html",
        )
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
