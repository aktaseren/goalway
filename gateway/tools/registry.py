"""
Tool registry .

Tools are functions exposed to models (via OpenAI-style tool calling)
and to MCP clients. Each tool declares:

- a JSON schema for its parameters,
- the RBAC permission needed to invoke it,
- an idempotency/safety hint for the model.

Tools never trust model output: arguments are validated against the
schema before the handler runs, and RBAC is enforced by the caller
(API layer / MCP server / agent orchestrator), not by the model.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("goalway.tools")


class ToolResult:
    """Uniform tool outcome: ok/error + JSON-serializable data."""

    def __init__(self, data: Any, ok: bool = True, error: str = ""):
        self.data = data
        self.ok = ok
        self.error = error

    def to_dict(self) -> dict:
        if self.ok:
            return {
                "ok": True,
                "data": self.data,
            }

        return {
            "ok": False,
            "error": self.error,
        }


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[[dict], Any],
        required_permission: str = "tools.call",
        read_only: bool = True,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.required_permission = required_permission
        self.read_only = read_only

    def to_openai(self) -> dict:
        """OpenAI function-tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp(self) -> dict:
        """MCP tools/list entry."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    def execute(self, arguments: dict) -> ToolResult:
        try:
            validated = self.validate_arguments(arguments)
            data = self.handler(**validated)
            return ToolResult(data)
        except ToolValidationError as exc:
            return ToolResult(None, ok=False, error=str(exc))
        except Exception as exc:
            logger.exception("tool %s failed", self.name)
            return ToolResult(None, ok=False, error=str(exc))

    def validate_arguments(self, arguments: dict) -> dict:
        if not isinstance(arguments, dict):
            raise ToolValidationError(
                f"Arguments for {self.name} must be an object"
            )

        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        for name in required:
            if name not in arguments:
                raise ToolValidationError(
                    f"Missing required argument: {name}"
                )

        # Drop unknown arguments so a confused model cannot smuggle
        # parameters into handlers.
        return {
            name: value
            for name, value in arguments.items()
            if name in properties
        }


class ToolValidationError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        # Instance-scoped store: a directly created ToolRegistry()
        # is isolated from the global registry returned by
        # get_tool_registry().
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda tool: tool.name)

    def openai_tools(self) -> list[dict]:
        return [tool.to_openai() for tool in self.list()]

    def mcp_tools(self) -> list[dict]:
        return [tool.to_mcp() for tool in self.list()]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        tool = self.get(name)

        if tool is None:
            return ToolResult(
                None,
                ok=False,
                error=f"Unknown tool: {name}",
            )

        return tool.execute(arguments)


_default_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _default_registry

    if _default_registry is None:
        _default_registry = ToolRegistry()

    return _default_registry


def __getattr__(name):
    """
    Backwards-compatible alias: gateway.tools.registry._REGISTRY is
    the global registry's store (kept alive for code that reads or
    cleans it directly).
    """
    if name == "_REGISTRY":
        return get_tool_registry()._tools

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def tool(
    name: str,
    description: str,
    parameters: dict,
    required_permission: str = "tools.call",
    read_only: bool = True,
):
    """Decorator to register a function as a tool."""

    def decorator(func: Callable):
        registry = get_tool_registry()
        registry.register(
            Tool(
                name=name,
                description=description,
                parameters=parameters,
                handler=func,
                required_permission=required_permission,
                read_only=read_only,
            )
        )
        return func

    return decorator
