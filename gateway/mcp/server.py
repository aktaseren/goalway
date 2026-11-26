"""
MCP (Model Context Protocol) server .

Implements the MCP JSON-RPC 2.0 message envelope with the core
server methods:

- initialize / notifications/initialized (capability handshake)
- tools/list                        (discover gateway tools)
- tools/call                        (execute a gateway tool)
- ping

Transport: stateless streamable HTTP -- a client POSTs a single
JSON-RPC message to /mcp and receives the JSON-RPC response as
plain JSON. This is the simplest spec-compliant shape for HTTP
transports and works with curl, the MCP inspector and most clients.

The gateway tool registry is the single source of truth for tools:
anything registered there is instantly available to both OpenAI-style
tool calling and MCP clients.
"""

import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from gateway.auth_context import AuthContext
from gateway.auth import get_auth_context
from gateway.config import (
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
)
from gateway.metrics import MCP_REQUESTS, TOOL_CALLS
from gateway.rbac import require_permission
from gateway.tools import get_tool_registry

logger = logging.getLogger("goalway.mcp")

PROTOCOL_VERSION = "2025-03-26"

mcp_router = APIRouter()


class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)


# Standard JSON-RPC / MCP error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _result(request_id, result: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _error(request_id, code: int, message: str, data=None) -> dict:
    error = {"code": code, "message": message}

    if data is not None:
        error["data"] = data

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }


def _handle_initialize(request_id: str, params: dict) -> dict:
    return _result(
        request_id,
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
            },
            "serverInfo": {
                "name": MCP_SERVER_NAME,
                "version": MCP_SERVER_VERSION,
            },
        },
    )


def _handle_tools_list(request_id: str) -> dict:
    registry = get_tool_registry()

    return _result(
        request_id,
        {"tools": registry.mcp_tools()},
    )


def _handle_tools_call(request_id: str, params: dict) -> dict:
    registry = get_tool_registry()

    name = params.get("name", "")
    arguments = params.get("arguments", {})

    tool = registry.get(name)

    if tool is None:
        raise JSONRPCError(
            METHOD_NOT_FOUND,
            f"Unknown tool: {name}",
        )

    result = registry.execute(name, arguments)

    TOOL_CALLS.labels(tool=name, source="mcp").inc()

    content = [
        {
            "type": "text",
            "text": json.dumps(
                result.to_dict(), default=str
            )[:16000],
        }
    ]

    return _result(
        request_id,
        {
            "content": content,
            "isError": not result.ok,
        },
    )


def handle_mcp_message(message: dict) -> dict:
    """
    Process one JSON-RPC 2.0 message and produce the response
    message (or a notification acknowledgement that callers may
    ignore).
    """
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, INVALID_REQUEST, "Not a JSON-RPC 2.0 message")

    method = message.get("method", "")
    request_id = message.get("id")
    params = message.get("params", {})

    # Notifications (no id) get no response per the JSON-RPC spec.
    if request_id is None and method:
        if method == "notifications/initialized":
            return {}
        return {}

    try:
        if method == "initialize":
            return _handle_initialize(request_id, params)

        if method == "tools/list":
            return _handle_tools_list(request_id)

        if method == "tools/call":
            return _handle_tools_call(request_id, params)

        if method == "ping":
            return _result(request_id, {})

        if method.startswith("notifications/"):
            return {}

        return _error(
            request_id,
            METHOD_NOT_FOUND,
            f"Unknown method: {method}",
        )

    except JSONRPCError as exc:
        return _error(request_id, exc.code, exc.message, exc.data)

    except Exception as exc:
        logger.exception("MCP internal error")
        return _error(request_id, INTERNAL_ERROR, str(exc))


@mcp_router.post("/mcp")
async def mcp_endpoint(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Stateless streamable-HTTP style MCP endpoint.

    One JSON-RPC message per POST. Batch arrays are not supported
    (single-message POSTs are the common client behavior).
    """
    require_permission(auth, "mcp.use")

    MCP_REQUESTS.inc()

    try:
        message = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            _error(None, PARSE_ERROR, "Invalid JSON"),
            status_code=200,
        )

    if isinstance(message, list):
        return JSONResponse(
            _error(None, INVALID_REQUEST, "Batch messages not supported"),
            status_code=200,
        )

    response = handle_mcp_message(message)

    # Notifications produce no response body.
    if not response:
        return JSONResponse({}, status_code=202)

    return JSONResponse(response)


@mcp_router.get("/mcp")
async def mcp_stream(
    auth: AuthContext = Depends(get_auth_context),
):
    """
    GET /mcp: server-initiated SSE streams are not used in the
    stateless mode; report capability per spec instead of 404.
    """
    require_permission(auth, "mcp.use")

    return JSONResponse(
        {
            "transport": "streamable-http",
            "mode": "stateless",
            "use": "POST JSON-RPC messages to /mcp",
        }
    )
