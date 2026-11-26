"""
Gateway self-introspection tools .

Let an agent answer questions about the platform it runs on:
which models are registered, what the gateway configuration is,
what tools exist.
"""

from gateway.tools.registry import tool


@tool(
    name="gateway_list_models",
    description=(
        "List the models registered in the AI gateway with their "
        "provider, health and enabled status."
    ),
    parameters={},
    required_permission="tools.call",
)
def gateway_list_models():
    from gateway.registry import list_models, get_model_config

    models = []

    for name in list_models():
        config = get_model_config(name) or {}

        models.append(
            {
                "model": name,
                "provider": config.get("provider", "unknown"),
                "enabled": config.get("enabled", False),
            }
        )

    return {"models": models}


@tool(
    name="gateway_list_tools",
    description=(
        "List the tools available in the gateway tool registry."
    ),
    parameters={},
    required_permission="tools.call",
)
def gateway_list_tools():
    from gateway.tools import get_tool_registry

    registry = get_tool_registry()

    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "read_only": t.read_only,
            }
            for t in registry.list()
        ]
    }
