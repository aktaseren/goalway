"""
Thin entrypoint alias: the Procfile (OpenShift S2I) starts `uvicorn app:app`

The Procfile (OpenShift S2I) starts `uvicorn app:app` while the
Dockerfile starts `uvicorn gateway.main:app`. Both now serve the
same application -- this file used to be a stale copy of the V4
monolith.
"""

from gateway.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
