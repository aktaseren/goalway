"""Tests for gateway/mcp/server.py (JSON-RPC MCP server)."""

import json
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gateway.api as api_module
import gateway.keys
from gateway.mcp.server import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    handle_mcp_message,
    mcp_router,
)
from gateway.rag.models import DocumentChunk, SearchResult
from gateway.tools import get_tool_registry

ADMIN_KEY = "redhat-demo-key"
USER_KEY = "team-user-key"


class FakeRAGService:
    def search(self, query: str, top_k: int = 5, **kwargs):
        return (
            [
                SearchResult(
                    chunk=DocumentChunk(
                        chunk_id="doc-1-0",
                        document_id="doc-1",
                        text="OpenShift AI provides model serving.",
                        position=0,
                    ),
                    score=0.95,
                )
            ][:top_k],
            SimpleNamespace(strategy="hybrid"),
        )


def msg(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def test_initialize_returns_server_info():
    response = handle_mcp_message(msg("initialize", {}))

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert response["result"]["serverInfo"]["name"] == (
        "goalway"
    )
    assert "version" in response["result"]["serverInfo"]
    assert "tools" in response["result"]["capabilities"]


def test_tools_list_returns_builtin_tools():
    response = handle_mcp_message(msg("tools/list"))

    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert "rag_search" in names
    assert "gateway_list_tools" in names

    # MCP tool shape.
    for tool in tools:
        assert "description" in tool
        assert "inputSchema" in tool


def test_tools_call_unknown_tool_method_not_found():
    response = handle_mcp_message(
        msg("tools/call", {"name": "no_such_tool", "arguments": {}})
    )

    assert "error" in response
    assert response["error"]["code"] == METHOD_NOT_FOUND
    assert response["id"] == 1


def test_tools_call_rag_search_ok(monkeypatch):
    monkeypatch.setattr(api_module, "rag_service", FakeRAGService())

    response = handle_mcp_message(
        msg("tools/call", {"name": "rag_search", "arguments": {"query": "x"}})
    )

    result = response["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"

    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is True
    assert payload["data"]["results"][0]["document_id"] == "doc-1"


def test_tools_call_invalid_arguments_is_error_result(monkeypatch):
    monkeypatch.setattr(api_module, "rag_service", FakeRAGService())

    response = handle_mcp_message(
        msg("tools/call", {"name": "rag_search", "arguments": {}})
    )

    # Validation failure is a tool-level error, not a JSON-RPC error.
    result = response["result"]
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is False
    assert "Missing required argument" in payload["error"]


def test_ping_returns_empty_result():
    response = handle_mcp_message(msg("ping"))

    assert response["result"] == {}
    assert "error" not in response


def test_invalid_jsonrpc_version_rejected():
    response = handle_mcp_message(
        {"jsonrpc": "1.0", "id": 1, "method": "ping"}
    )

    assert response["error"]["code"] == INVALID_REQUEST


def test_non_object_message_rejected():
    response = handle_mcp_message("not-a-message")

    assert response["error"]["code"] == INVALID_REQUEST


def test_unknown_method_rejected():
    response = handle_mcp_message(msg("resources/list"))

    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_notification_returns_falsy():
    response = handle_mcp_message(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}
    )

    assert response == {}

    response = handle_mcp_message({"jsonrpc": "2.0", "method": "anything"})

    assert response == {}


# ---------------------------------------------------------------
# /mcp HTTP endpoint (auth + RBAC)
# ---------------------------------------------------------------

@pytest.fixture
def mcp_client():
    # Make sure the real key store is loaded for API-key auth.
    gateway.keys.load_api_keys(
        os.path.join(ROOT, "config", "api_keys.json")
    )

    app = FastAPI()
    app.include_router(mcp_router)

    with TestClient(app) as client:
        yield client


def test_mcp_endpoint_initialize(mcp_client):
    response = mcp_client.post(
        "/mcp",
        json=msg("initialize", {}),
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert body["result"]["serverInfo"]["name"] == "goalway"


def test_mcp_endpoint_tools_list(mcp_client):
    response = mcp_client.post(
        "/mcp",
        json=msg("tools/list"),
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 200
    names = {t["name"] for t in response.json()["result"]["tools"]}
    assert "rag_search" in names


def test_mcp_endpoint_user_key_forbidden(mcp_client):
    response = mcp_client.post(
        "/mcp",
        json=msg("initialize", {}),
        headers={"X-API-Key": USER_KEY},
    )

    assert response.status_code == 403


def test_mcp_endpoint_missing_auth(mcp_client):
    response = mcp_client.post("/mcp", json=msg("initialize", {}))

    assert response.status_code == 401


def test_mcp_endpoint_notification_accepted(mcp_client):
    response = mcp_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 202
    assert response.json() == {}


def test_mcp_endpoint_get_describes_transport(mcp_client):
    response = mcp_client.get("/mcp", headers={"X-API-Key": ADMIN_KEY})

    assert response.status_code == 200
    assert response.json()["transport"] == "streamable-http"


def test_mcp_endpoint_batch_rejected(mcp_client):
    response = mcp_client.post(
        "/mcp",
        json=[msg("initialize", {}), msg("ping", request_id=2)],
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 200
    assert response.json()["error"]["code"] == INVALID_REQUEST
