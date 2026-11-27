"""Tests for gateway/llm.py (programmatic completion + tool loop)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import gateway.llm
import gateway.registry
import gateway.tools.registry as tool_registry_module
from gateway.llm import LLMError, complete, first_message, resolve_model
from gateway.tools import get_tool_registry
from gateway.tools.registry import Tool

MODEL_REGISTRY_FILE = os.path.join(ROOT, "config", "models.json")


@pytest.fixture(scope="module", autouse=True)
def loaded_registry():
    """Ensure the model registry is populated for this module."""
    previous = dict(gateway.registry.MODEL_REGISTRY)
    gateway.registry.load_model_registry(MODEL_REGISTRY_FILE)
    yield
    gateway.registry.MODEL_REGISTRY.clear()
    gateway.registry.MODEL_REGISTRY.update(previous)


# ---------------------------------------------------------------
# resolve_model / first_message / complete
# ---------------------------------------------------------------

def test_resolve_model_unknown_raises():
    with pytest.raises(LLMError):
        resolve_model("definitely-not-a-model")


def test_resolve_model_disabled_raises():
    gateway.registry.MODEL_REGISTRY["disabled-model"] = {
        "enabled": False,
    }

    try:
        with pytest.raises(LLMError):
            resolve_model("disabled-model")
    finally:
        del gateway.registry.MODEL_REGISTRY["disabled-model"]


def test_resolve_model_known_healthy(monkeypatch):
    monkeypatch.setitem(gateway.registry.MODEL_HEALTH, "echo", True)

    model, model_config = resolve_model("echo")

    assert model == "echo"
    assert model_config["enabled"] is True


def test_resolve_model_auto_routes_to_echo(monkeypatch):
    monkeypatch.setitem(gateway.registry.MODEL_HEALTH, "echo", True)

    model, _ = resolve_model("auto", "hello")

    assert model == "echo"


def test_resolve_model_unhealthy_falls_over_to_local_rag():
    # No health entries -> granite considered unhealthy; the local
    # rag model has no backend to probe and is always healthy, so it
    # is the failover target (previously this raised LLMError).
    model, config = resolve_model("granite")

    assert model == "rag"
    assert config["provider"] == "rag"


def test_first_message_extracts_message():
    data = {
        "choices": [
            {"message": {"role": "assistant", "content": "hi"}},
        ]
    }

    assert first_message(data) == {"role": "assistant", "content": "hi"}


def test_first_message_bad_shape_raises():
    with pytest.raises(LLMError):
        first_message({})

    with pytest.raises(LLMError):
        first_message({"choices": []})

    with pytest.raises(LLMError):
        first_message({"choices": [{"no_message": True}]})


def test_complete_raises_llm_error_on_bad_backend_shape(monkeypatch):
    monkeypatch.setattr(
        gateway.llm,
        "resolve_model",
        lambda model, prompt="": ("echo", {"backends": ["http://x"]}),
    )
    monkeypatch.setattr(
        gateway.llm, "get_backend", lambda model, backends: backends[0]
    )

    class FakeResponse:
        @staticmethod
        def json():
            return {"unexpected": "shape"}

    monkeypatch.setattr(
        gateway.llm,
        "post_with_retries",
        lambda **kwargs: (FakeResponse(), 0),
    )

    with pytest.raises(LLMError, match="Unexpected backend response shape"):
        complete(model="echo", messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------
# complete_with_tools (agentic loop)
# ---------------------------------------------------------------

@pytest.fixture
def weather_tool():
    """Register a temporary tool on the global registry, then remove it."""
    calls = []

    def lookup(city: str):
        calls.append(city)
        return {"city": city, "forecast": "sunny"}

    get_tool_registry().register(
        Tool(
            name="test_weather_lookup",
            description="Look up the weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
            handler=lookup,
        )
    )

    yield calls

    tool_registry_module._REGISTRY.pop("test_weather_lookup", None)


def tool_call_response(name: str, arguments: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            }
        ]
    }


def final_message(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_complete_with_tools_executes_tool_then_answers(
    monkeypatch, weather_tool
):
    state = {"calls": 0}

    def fake_complete(model, messages, tools=None, temperature=None,
                      max_tokens=None):
        state["calls"] += 1

        if state["calls"] == 1:
            # First round: the model asks for a tool.
            return tool_call_response(
                "test_weather_lookup",
                '{"city": "Boston"}',
            )

        # Second round: final answer.
        return final_message("It is sunny in Boston.")

    monkeypatch.setattr(gateway.llm, "complete", fake_complete)

    result = gateway.llm.complete_with_tools(
        model="echo",
        messages=[{"role": "user", "content": "weather in Boston?"}],
        tools=[{"type": "function", "function": {"name": "irrelevant"}}],
    )

    # The tool handler actually ran.
    assert weather_tool == ["Boston"]

    assert result["answer"] == "It is sunny in Boston."
    assert result["iterations"] == 2
    assert "stopped_early" not in result

    # Tool call record shape.
    assert len(result["tool_calls"]) == 1
    record = result["tool_calls"][0]
    assert record["tool"] == "test_weather_lookup"
    assert record["arguments"] == {"city": "Boston"}
    assert record["status"] == "ok"
    assert record["iteration"] == 1
    assert "latency_ms" in record


def test_complete_with_tools_no_tool_calls(monkeypatch):
    monkeypatch.setattr(
        gateway.llm,
        "complete",
        lambda **kwargs: final_message("Direct answer."),
    )

    result = gateway.llm.complete_with_tools(
        model="echo",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
    )

    assert result["answer"] == "Direct answer."
    assert result["tool_calls"] == []
    assert result["iterations"] == 1


def test_complete_with_tools_budget_exhaustion(monkeypatch, weather_tool):
    # Model keeps requesting the tool forever -> budget exhausted.
    monkeypatch.setattr(
        gateway.llm,
        "complete",
        lambda model, messages, tools=None, temperature=None,
        max_tokens=None: tool_call_response(
            "test_weather_lookup", '{"city": "Boston"}'
        ),
    )

    result = gateway.llm.complete_with_tools(
        model="echo",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[],
        max_iterations=2,
    )

    assert result["iterations"] == 2
    assert result["stopped_early"] is True
    assert len(result["tool_calls"]) == 2
    assert len(weather_tool) == 2


def test_complete_with_tools_bad_arguments_json(monkeypatch, weather_tool):
    monkeypatch.setattr(
        gateway.llm,
        "complete",
        lambda model, messages, tools=None, temperature=None,
        max_tokens=None: tool_call_response(
            "test_weather_lookup", "{not valid json"
        ),
    )

    result = gateway.llm.complete_with_tools(
        model="echo",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[],
        max_iterations=1,
    )

    # Unparseable arguments become {} -> required arg missing -> the
    # tool result is an error record, but the loop keeps going.
    assert result["tool_calls"][0]["arguments"] == {}
    assert weather_tool == []  # handler never invoked
