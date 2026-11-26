"""
GitHub tools .

Read-only GitHub API access for agents: list pull requests, read
issues, check workflow runs. Requires GITHUB_TOKEN (mounted from a
Secret in OpenShift) for private repos and higher rate limits.
"""

import logging

import requests

from gateway.config import GITHUB_API_URL, GITHUB_TOKEN
from gateway.tools.registry import tool

logger = logging.getLogger("goalway.tools.github")

REQUEST_TIMEOUT_SECONDS = 15


def _github_get(path: str, params: dict | None = None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}

    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    response = requests.get(
        f"{GITHUB_API_URL.rstrip('/')}{path}",
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 403:
        return {
            "error": "Forbidden (rate limit or missing GITHUB_TOKEN)",
        }

    response.raise_for_status()

    return response.json()


def _parse_repo(repository: str) -> tuple[str, str]:
    if "/" not in repository:
        raise ValueError(
            "repository must be in owner/name form, e.g. 'aktaseren/goalway'"
        )

    owner, name = repository.split("/", 1)

    return owner.strip(), name.strip()


@tool(
    name="github_list_pull_requests",
    description=(
        "List pull requests in a GitHub repository. Optionally "
        "filter by state (open/closed/all)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository in owner/name form",
            },
            "state": {
                "type": "string",
                "enum": ["open", "closed", "all"],
                "description": "PR state filter (default open)",
            },
        },
        "required": ["repository"],
    },
    required_permission="tools.call",
)
def github_list_pull_requests(repository: str, state: str = "open"):
    owner, name = _parse_repo(repository)

    data = _github_get(
        f"/repos/{owner}/{name}/pulls",
        params={"state": state, "per_page": 20},
    )

    if isinstance(data, dict) and "error" in data:
        return data

    pulls = [
        {
            "number": pr["number"],
            "title": pr["title"],
            "user": pr["user"]["login"],
            "state": pr["state"],
            "draft": pr.get("draft", False),
        }
        for pr in data
    ]

    return {"repository": f"{owner}/{name}", "pull_requests": pulls}


@tool(
    name="github_get_issue",
    description="Fetch a single GitHub issue by number.",
    parameters={
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository in owner/name form",
            },
            "number": {
                "type": "integer",
                "description": "Issue number",
            },
        },
        "required": ["repository", "number"],
    },
    required_permission="tools.call",
)
def github_get_issue(repository: str, number: int):
    owner, name = _parse_repo(repository)

    data = _github_get(f"/repos/{owner}/{name}/issues/{number}")

    if isinstance(data, dict) and "error" in data:
        return data

    return {
        "number": data["number"],
        "title": data["title"],
        "state": data["state"],
        "author": data["user"]["login"],
        "body": (data.get("body") or "")[:2000],
        "labels": [label["name"] for label in data.get("labels", [])],
    }


@tool(
    name="github_list_workflow_runs",
    description=(
        "List recent GitHub Actions workflow runs for a repository."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository in owner/name form",
            },
        },
        "required": ["repository"],
    },
    required_permission="tools.call",
)
def github_list_workflow_runs(repository: str):
    owner, name = _parse_repo(repository)

    data = _github_get(
        f"/repos/{owner}/{name}/actions/runs",
        params={"per_page": 10},
    )

    if isinstance(data, dict) and "error" in data:
        return data

    runs = [
        {
            "name": run["name"],
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "branch": run.get("head_branch"),
            "created_at": run.get("created_at"),
        }
        for run in data.get("workflow_runs", [])
    ]

    return {"repository": f"{owner}/{name}", "workflow_runs": runs}
