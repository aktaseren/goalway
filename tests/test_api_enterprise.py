"""
End-to-end API tests for the features on the real gateway app.

All network-touching paths are mocked; the tests exercise auth
(API keys + roles), admin key management, feedback, MCP, quotas and
the tool-calling chat flow through the actual FastAPI routes.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from fastapi.testclient import TestClient

import gateway.api as api_module
import gateway.keys
import gateway.llm
import gateway.rag.feedback as feedback_module
from gateway.keys import QuotaExceededError
from gateway.main import app

ADMIN_KEY = "redhat-demo-key"
USER_KEY = "team-user-key"

client = TestClient(app)


@pytest.fixture(autouse=True)
def real_key_store():
    """Point the key store at the repo's config/api_keys.json."""
    gateway.keys.load_api_keys(
        os.path.join(ROOT, "config", "api_keys.json")
    )
    yield


@pytest.fixture
def clean_feedback_store():
    feedback_module._memory_store.clear()
    yield
    feedback_module._memory_store.clear()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "models" in body


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200


def test_admin_keys_lists_all_keys():
    response = client.get("/admin/keys", headers={"X-API-Key": ADMIN_KEY})

    assert response.status_code == 200
    keys = response.json()["keys"]
    assert len(keys) == 3

    roles = {entry["role"] for entry in keys}
    assert roles == {"admin", "user"}

    # Secrets are never exposed.
    assert ADMIN_KEY not in response.text
    assert USER_KEY not in response.text


def test_admin_keys_forbidden_for_user_role():
    response = client.get("/admin/keys", headers={"X-API-Key": USER_KEY})

    assert response.status_code == 403
    assert response.json()["detail"]["permission"] == "admin.reload"


def test_admin_keys_requires_auth():
    response = client.get("/admin/keys")

    assert response.status_code == 401


def test_admin_quota_endpoint():
    response = client.get("/admin/quota", headers={"X-API-Key": ADMIN_KEY})

    assert response.status_code == 200
    usage = response.json()["usage"]
    assert "admin" in usage
    assert usage["admin"]["limit"] >= 1
    assert "used" in usage["admin"]


def test_rag_feedback_recorded_for_user(clean_feedback_store):
    response = client.post(
        "/v1/rag/feedback",
        json={
            "rating": 1,
            "query": "how do pods work",
            "document_id": "doc-1",
            "comment": "helpful",
        },
        headers={"X-API-Key": USER_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "recorded"
    assert body["feedback_id"].startswith("fb-")


def test_rag_feedback_rejects_zero_rating(clean_feedback_store):
    response = client.post(
        "/v1/rag/feedback",
        json={"rating": 0},
        headers={"X-API-Key": USER_KEY},
    )

    assert response.status_code == 400


def test_rag_feedback_stats_after_record(clean_feedback_store):
    client.post(
        "/v1/rag/feedback",
        json={"rating": 1, "query": "pods", "document_id": "doc-1"},
        headers={"X-API-Key": USER_KEY},
    )

    response = client.get(
        "/v1/rag/feedback/stats", headers={"X-API-Key": USER_KEY}
    )

    assert response.status_code == 200
    stats = response.json()
    assert stats["total_feedback"] == 1
    assert stats["positive"] == 1
    assert stats["satisfaction_rate"] == 1.0


def test_rag_groundedness_without_query(clean_feedback_store):
    response = client.post(
        "/v1/rag/groundedness",
        json={"answer": "The gateway runs on Mars and costs five dollars."},
        headers={"X-API-Key": USER_KEY},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["verdict"] == "likely_hallucinated"
    assert report["documents_checked"] == []
    assert report["unsupported_claims"]


def test_mcp_initialize_with_admin_key():
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        },
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["serverInfo"]["name"] == "goalway"
    assert body["result"]["protocolVersion"] == "2025-03-26"


def test_mcp_forbidden_for_user_key():
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        headers={"X-API-Key": USER_KEY},
    )

    assert response.status_code == 403


def test_chat_quota_exceeded_returns_429(monkeypatch):
    # Neutralize the per-minute rate limit for this test.
    monkeypatch.setattr(api_module, "check_rate_limit", lambda *a, **k: None)
    monkeypatch.setattr(api_module, "QUOTA_ENABLED", True)

    def raise_quota(api_key):
        raise QuotaExceededError(
            key_name="admin", limit=1, reset_in_seconds=60
        )

    monkeypatch.setattr(api_module, "check_quota", raise_quota)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["error"] == "Quota exceeded"
    assert detail["limit"] == 1
    assert detail["retry_after_seconds"] == 60


def test_chat_with_tools_runs_tool_loop(monkeypatch):
    monkeypatch.setattr(api_module, "check_rate_limit", lambda *a, **k: None)

    fake_result = {
        "answer": "The gateway has 3 registered models.",
        "tool_calls": [
            {
                "tool": "gateway_list_models",
                "arguments": {},
                "status": "ok",
                "latency_ms": 1.5,
                "iteration": 1,
            }
        ],
        "iterations": 1,
    }

    monkeypatch.setattr(
        gateway.llm, "complete_with_tools", lambda **kwargs: fake_result
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "what models are available?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "gateway_list_models",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["choices"][0]["message"]["content"] == (
        "The gateway has 3 registered models."
    )
    assert body["choices"][0]["finish_reason"] == "stop"

    gateway_info = body["gateway"]
    assert gateway_info["tool_calls"][0]["tool"] == "gateway_list_models"
    assert gateway_info["iterations"] == 1
    assert gateway_info["stopped_early"] is False


def test_chat_unknown_model_returns_400(monkeypatch):
    monkeypatch.setattr(api_module, "check_rate_limit", lambda *a, **k: None)
    monkeypatch.setattr(api_module, "QUOTA_ENABLED", False)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "not-a-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == 400
    assert "Unknown or disabled model" in response.json()["detail"]["error"]


def test_chat_requires_permission():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 401
