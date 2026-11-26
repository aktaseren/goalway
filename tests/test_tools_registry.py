"""Tests for gateway/tools/registry.py + registered built-in tools ."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from types import SimpleNamespace

import gateway.api as api_module
import gateway.tools.registry as registry_module
from gateway.rag.models import DocumentChunk, SearchResult
from gateway.tools import get_tool_registry
from gateway.tools.registry import (
    Tool,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
)


def cleanup_tool(name: str):
    """Remove a test-only tool from the shared registry dict."""
    registry_module._REGISTRY.pop(name, None)


def make_tool(name="test_multiply", handler=None, **kwargs):
    return Tool(
        name=name,
        description="Multiply numbers for tests",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a"],
        },
        handler=handler or (lambda a, b=2: a * b),
        **kwargs,
    )


def test_register_and_execute_custom_tool():
    registry = ToolRegistry()
    registry.register(make_tool())

    try:
        result = registry.execute("test_multiply", {"a": 3, "b": 4})

        assert isinstance(result, ToolResult)
        assert result.ok is True
        assert result.data == 12
        assert result.to_dict() == {"ok": True, "data": 12}
    finally:
        cleanup_tool("test_multiply")


def test_execute_missing_required_argument():
    registry = ToolRegistry()
    registry.register(make_tool())

    try:
        result = registry.execute("test_multiply", {"b": 4})

        assert result.ok is False
        assert "Missing required argument" in result.error
        assert result.to_dict() == {"ok": False, "error": result.error}
    finally:
        cleanup_tool("test_multiply")


def test_execute_non_object_arguments_rejected():
    registry = ToolRegistry()
    registry.register(make_tool())

    try:
        result = registry.execute("test_multiply", "not-a-dict")

        assert result.ok is False
        assert result.error
    finally:
        cleanup_tool("test_multiply")


def test_execute_unknown_arguments_dropped():
    def handler(**kwargs):
        return kwargs

    registry = ToolRegistry()
    registry.register(make_tool(name="test_kwargs", handler=handler))

    try:
        result = registry.execute(
            "test_kwargs",
            {"a": 1, "b": 5, "sneaky": "must be dropped"},
        )

        assert result.ok is True
        assert result.data == {"a": 1, "b": 5}
    finally:
        cleanup_tool("test_kwargs")


def test_execute_handler_error_returns_not_ok():
    def boom(a, b=2):
        raise RuntimeError("handler exploded")

    registry = ToolRegistry()
    registry.register(make_tool(name="test_boom", handler=boom))

    try:
        result = registry.execute("test_boom", {"a": 1})

        assert result.ok is False
        assert "handler exploded" in result.error
    finally:
        cleanup_tool("test_boom")


def test_execute_unknown_tool():
    result = ToolRegistry().execute("no_such_tool", {})

    assert result.ok is False
    assert "Unknown tool" in result.error


def test_validate_arguments_raises_for_missing():
    tool = make_tool()

    with pytest.raises(ToolValidationError):
        tool.validate_arguments({"b": 1})


def test_openai_schema_shape():
    tool = make_tool()

    schema = tool.to_openai()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "test_multiply"
    assert schema["function"]["description"]
    assert schema["function"]["parameters"]["required"] == ["a"]


def test_mcp_schema_shape():
    tool = make_tool()

    schema = tool.to_mcp()

    assert schema["name"] == "test_multiply"
    assert schema["description"]
    assert schema["inputSchema"]["type"] == "object"
    assert "properties" in schema["inputSchema"]


def test_registry_list_and_schemas():
    registry = ToolRegistry()
    registry.register(make_tool(name="test_zzz"))

    try:
        tools = registry.list()
        assert any(t.name == "test_zzz" for t in tools)

        openai_tools = registry.openai_tools()
        assert all(t["type"] == "function" for t in openai_tools)

        mcp_tools = registry.mcp_tools()
        assert all("inputSchema" in t for t in mcp_tools)
    finally:
        cleanup_tool("test_zzz")


def test_builtin_tools_registered():
    # Importing gateway.tools registers the built-ins.
    import gateway.tools  # noqa: F401

    names = {tool.name for tool in get_tool_registry().list()}

    assert "rag_search" in names
    assert "rag_ingest" in names
    assert "k8s_list_pods" in names
    assert "k8s_list_deployments" in names
    assert "gateway_list_models" in names
    assert "gateway_list_tools" in names


def test_gateway_list_tools_executes():
    result = get_tool_registry().execute("gateway_list_tools", {})

    assert result.ok is True
    tool_names = {t["name"] for t in result.data["tools"]}
    assert "rag_search" in tool_names
    assert "gateway_list_tools" in tool_names


def test_gateway_list_models_executes():
    result = get_tool_registry().execute("gateway_list_models", {})

    assert result.ok is True
    assert "models" in result.data


class FakeRAGService:
    def search(self, query: str, top_k: int = 5, **kwargs):
        results = [
            SearchResult(
                chunk=DocumentChunk(
                    chunk_id="doc-1-0",
                    document_id="doc-1",
                    text="OpenShift AI provides enterprise model serving.",
                    position=0,
                ),
                score=0.95,
            )
        ][:top_k]
        return results, SimpleNamespace(strategy="hybrid")

    def ingest(self, document):
        return 3


def test_rag_search_tool_with_fake_service(monkeypatch):
    monkeypatch.setattr(
        api_module, "rag_service", FakeRAGService()
    )

    result = get_tool_registry().execute(
        "rag_search", {"query": "model serving", "top_k": 3}
    )

    assert result.ok is True
    assert result.data["total"] == 1
    assert result.data["results"][0]["document_id"] == "doc-1"
    assert result.data["results"][0]["chunk_id"] == "doc-1-0"
    assert result.data["results"][0]["score"] == 0.95


def test_rag_ingest_tool_with_fake_service(monkeypatch):
    monkeypatch.setattr(
        api_module, "rag_service", FakeRAGService()
    )

    result = get_tool_registry().execute(
        "rag_ingest",
        {"document_id": "doc-9", "text": "Some document text"},
    )

    assert result.ok is True
    assert result.data == {
        "document_id": "doc-9",
        "chunks_created": 3,
    }


def test_rag_search_requires_query():
    result = get_tool_registry().execute("rag_search", {})

    assert result.ok is False
    assert "Missing required argument" in result.error
