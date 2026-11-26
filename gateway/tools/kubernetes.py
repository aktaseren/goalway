"""
Kubernetes / OpenShift tools .

Read-only cluster introspection through the Kubernetes API. Inside
OpenShift the gateway authenticates with its service account token;
set KUBERNETES_API_URL to target another cluster (e.g. the CRC
cluster API from a laptop).
"""

import logging
import os

import requests

from gateway.config import KUBERNETES_API_URL
from gateway.tools.registry import tool

logger = logging.getLogger("goalway.tools.k8s")

REQUEST_TIMEOUT_SECONDS = 15

_namespace: str | None = None


def _api_url() -> str:
    if KUBERNETES_API_URL:
        return KUBERNETES_API_URL

    host = os.getenv("KUBERNETES_SERVICE_HOST")

    if host:
        port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
        return f"https://{host}:{port}"

    return ""


def _session() -> requests.Session | None:
    """
    Build an authenticated session. Returns None when running
    outside a cluster without KUBERNETES_API_URL configured.
    """
    api_url = _api_url()

    if not api_url:
        return None

    session = requests.Session()
    session.verify = False  # in-cluster CA handling below

    token_file = "/var/run/secrets/kubernetes.io/serviceaccount/token"

    if os.path.exists(token_file):
        with open(token_file) as f:
            session.headers["Authorization"] = f"Bearer {f.read().strip()}"

        ca_file = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        if os.path.exists(ca_file):
            session.verify = ca_file
    else:
        session.headers["Authorization"] = "Bearer no-token"

    session.headers["Accept"] = "application/json"
    session.base_url = api_url.rstrip("/")

    return session


def _namespace_name() -> str:
    global _namespace

    if _namespace:
        return _namespace

    ns_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

    if os.path.exists(ns_file):
        with open(ns_file) as f:
            _namespace = f.read().strip()
    else:
        _namespace = "default"

    return _namespace


def _get(path: str) -> dict:
    session = _session()

    if session is None:
        return {
            "error": "Kubernetes API not configured",
            "hint": "Set KUBERNETES_API_URL or run in-cluster",
        }

    response = session.get(
        f"{session.base_url}{path}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 403:
        return {
            "error": "Forbidden: the gateway service account lacks permission",
            "status": 403,
        }

    response.raise_for_status()

    return response.json()


@tool(
    name="k8s_list_pods",
    description=(
        "List pods in a namespace, optionally filtered by label "
        "selector. Returns name, phase and restart counts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Kubernetes namespace (defaults to the gateway's own)",
            },
            "label_selector": {
                "type": "string",
                "description": 'Label selector, e.g. "app=gateway"',
            },
        },
    },
    required_permission="tools.call",
)
def k8s_list_pods(namespace: str = "", label_selector: str = ""):
    namespace = namespace or _namespace_name()

    path = f"/api/v1/namespaces/{namespace}/pods"

    if label_selector:
        path += f"?labelSelector={label_selector}"

    data = _get(path)

    if "error" in data:
        return data

    pods = []

    for item in data.get("items", []):
        status = item.get("status", {})

        pods.append(
            {
                "name": item["metadata"]["name"],
                "phase": status.get("phase"),
                "restarts": sum(
                    container.get("restartCount", 0)
                    for container in status.get("containerStatuses", [])
                ),
                "ready": sum(
                    1
                    for container in status.get("containerStatuses", [])
                    if container.get("ready")
                ),
            }
        )

    return {"namespace": namespace, "pods": pods, "total": len(pods)}


@tool(
    name="k8s_list_deployments",
    description=(
        "List deployments in a namespace with replica status."
    ),
    parameters={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Kubernetes namespace (defaults to the gateway's own)",
            },
        },
    },
    required_permission="tools.call",
)
def k8s_list_deployments(namespace: str = ""):
    namespace = namespace or _namespace_name()

    data = _get(f"/apis/apps/v1/namespaces/{namespace}/deployments")

    if "error" in data:
        return data

    deployments = []

    for item in data.get("items", []):
        status = item.get("status", {})

        deployments.append(
            {
                "name": item["metadata"]["name"],
                "replicas": status.get("replicas", 0),
                "ready": status.get("readyReplicas", 0),
                "available": status.get("availableReplicas", 0),
            }
        )

    return {
        "namespace": namespace,
        "deployments": deployments,
    }


@tool(
    name="k8s_get_events",
    description=(
        "List recent warning events in a namespace (crashes, "
        "failed probes, scheduling problems)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Kubernetes namespace (defaults to the gateway's own)",
            },
        },
    },
    required_permission="tools.call",
)
def k8s_get_events(namespace: str = ""):
    namespace = namespace or _namespace_name()

    data = _get(f"/api/v1/namespaces/{namespace}/events")

    if "error" in data:
        return data

    warnings = [
        {
            "object": event.get("involvedObject", {}).get("name"),
            "reason": event.get("reason"),
            "message": event.get("message"),
            "count": event.get("count", 1),
        }
        for event in data.get("items", [])
        if event.get("type") == "Warning"
    ]

    return {"namespace": namespace, "warnings": warnings}
