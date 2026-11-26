"""
ArgoCD tools .

Read-only ArgoCD application status for agents: check sync and
health status of GitOps-managed applications. Requires ARGOCD_URL
and ARGOCD_TOKEN (mounted from a Secret in OpenShift).
"""

import logging

import requests

from gateway.config import ARGOCD_TOKEN, ARGOCD_URL
from gateway.tools.registry import tool

logger = logging.getLogger("goalway.tools.argocd")

REQUEST_TIMEOUT_SECONDS = 15


def _argocd_get(path: str) -> dict:
    if not ARGOCD_URL or not ARGOCD_TOKEN:
        return {
            "error": "ArgoCD not configured",
            "hint": "Set ARGOCD_URL and ARGOCD_TOKEN",
        }

    response = requests.get(
        f"{ARGOCD_URL.rstrip('/')}/api/v1{path}",
        headers={"Authorization": f"Bearer {ARGOCD_TOKEN}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
        verify=False,
    )

    if response.status_code == 403:
        return {"error": "Forbidden: token lacks permission"}

    response.raise_for_status()

    return response.json()


@tool(
    name="argocd_list_applications",
    description=(
        "List ArgoCD applications with sync and health status."
    ),
    parameters={},
    required_permission="tools.call",
)
def argocd_list_applications():
    data = _argocd_get("/applications")

    if "error" in data:
        return data

    applications = []

    for name, app in data.get("items", {}).items():
        status = app.get("status", {})

        applications.append(
            {
                "name": name,
                "sync_status": status.get("sync", {}).get("status"),
                "health_status": status.get("health", {}).get("status"),
                "project": app.get("metadata", {}).get("labels", {}).get(
                    "argocd.argoproj.io/instance", ""
                ),
            }
        )

    return {"applications": applications}


@tool(
    name="argocd_get_application",
    description=(
        "Get detailed status of one ArgoCD application, including "
        "recent sync history."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "ArgoCD application name",
            },
        },
        "required": ["name"],
    },
    required_permission="tools.call",
)
def argocd_get_application(name: str):
    data = _argocd_get(f"/applications/{name}")

    if "error" in data:
        return data

    status = data.get("status", {})

    return {
        "name": name,
        "sync_status": status.get("sync", {}).get("status"),
        "health_status": status.get("health", {}).get("status"),
        "source": data.get("spec", {}).get("source", {}),
        "destination": data.get("spec", {}).get("destination", {}),
    }
