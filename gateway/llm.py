"""
Programmatic LLM completion client .

Used internally by the RAG judge, the agent orchestrator and the
tool-calling loop. Unlike the /v1/chat/completions HTTP path it has
no conversation side effects, no caching and no rate limits -- it is
the gateway calling a model on its own behalf.

Resolves models through the same model registry and provider plugins
as external traffic, so `model="auto"`, failover and health checks
all keep working.
"""

import logging
import time
import uuid

import requests

from gateway.config import (
    AGENT_MAX_ITERATIONS,
)
from gateway.registry import (
    get_model_config,
    is_model_healthy,
)
from gateway.failover import get_failover_model
from gateway.loadbalancer import get_backend
from gateway.resilience import post_with_retries
from gateway.model_selection import choose_model

logger = logging.getLogger("goalway.llm")


class LLMError(Exception):
    pass


def resolve_model(model: str, prompt: str = "") -> tuple[str, dict]:
    """
    Resolve a model name to (model, model_config) applying auto
    routing and health-aware failover.
    """
    if model == "auto":
        model = choose_model(prompt)

    model_config = get_model_config(model)

    if not model_config or not model_config.get("enabled", False):
        raise LLMError(f"Unknown or disabled model: {model}")

    if not is_model_healthy(model):
        failover_model = get_failover_model(model)

        if not failover_model:
            raise LLMError(f"Model unhealthy with no fallback: {model}")

        model = failover_model
        model_config = get_model_config(model)

    return model, model_config


def complete(
    model: str,
    messages: list,
    tools: list | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Run one chat completion against a registered backend.

    Returns the parsed, OpenAI-shaped response dict:
    {"choices": [{"message": {"role", "content", ...}}], ...}
    """
    prompt = ""

    for message in messages:
        if message.get("role") == "user":
            prompt = message.get("content", "")
            break

    model, model_config = resolve_model(model, prompt)

    backends = model_config.get("backends") or [model_config.get("url")]

    backend_url = get_backend(model, backends)

    if not backend_url:
        raise LLMError(f"Model '{model}' has no backend URL configured")

    payload = {
        "model": model,
        "messages": messages,
    }

    if tools:
        payload["tools"] = tools

    if temperature is not None:
        payload["temperature"] = temperature

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        response, retry_count = post_with_retries(
            url=backend_url,
            payload=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
            max_retries=1,
        )
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"Backend unavailable: {exc}") from exc

    data = response.json()

    if "choices" not in data:
        raise LLMError(f"Unexpected backend response shape: {list(data)}")

    return data


def first_message(data: dict) -> dict:
    """Extract the first choice's message from a completion."""
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("No message in completion response") from exc


def complete_text(
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
) -> str:
    """Convenience: single prompt -> plain text answer."""
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    data = complete(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return first_message(data).get("content", "")


def complete_with_tools(
    model: str,
    messages: list,
    tools: list,
    max_iterations: int | None = None,
) -> dict:
    """
    Agentic tool-calling loop .

    Runs a chat completion with tools enabled. When the model responds
    with tool_calls, executes them through the gateway tool registry,
    appends the results and iterates -- until the model produces a
    final answer or the iteration budget is exhausted.

    Returns {"answer": str, "tool_calls": [...], "iterations": int}.
    """
    from gateway.tools import get_tool_registry

    registry = get_tool_registry()
    iterations = 0
    all_tool_calls: list[dict] = []
    conversation = list(messages)

    budget = max_iterations or AGENT_MAX_ITERATIONS

    while iterations < budget:
        iterations += 1

        data = complete(
            model=model,
            messages=conversation,
            tools=tools,
        )

        message = first_message(data)

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            return {
                "answer": message.get("content", ""),
                "tool_calls": all_tool_calls,
                "iterations": iterations,
            }

        # Append the assistant's tool-call message, then the results.
        conversation.append(message)

        for tool_call in tool_calls:
            import json

            tool_call_id = tool_call.get("id", str(uuid.uuid4()))
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            arguments = function.get("arguments", "{}")

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

            started = time.time()

            try:
                result = registry.execute(tool_name, arguments)
                status = "ok"
            except Exception as exc:
                result = {"error": str(exc)}
                status = "error"

            elapsed_ms = round((time.time() - started) * 1000, 2)

            record = {
                "tool": tool_name,
                "arguments": arguments,
                "status": status,
                "latency_ms": elapsed_ms,
                "iteration": iterations,
            }
            all_tool_calls.append(record)

            logger.info("tool_call tool=%s status=%s latency_ms=%s",
                        tool_name, status, elapsed_ms)

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(
                        result if isinstance(result, dict) else {"result": str(result)}
                    )[:8000],
                }
            )

    # Budget exhausted: force a final answer without tools.
    data = complete(model=model, messages=conversation)
    message = first_message(data)

    return {
        "answer": message.get("content", ""),
        "tool_calls": all_tool_calls,
        "iterations": iterations,
        "stopped_early": True,
    }
