"""Tests for gateway/model_selection.py ('auto' model routing)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.model_selection import choose_model


def test_greeting_routes_to_echo():
    for greeting in ("hi", "hello", "hey"):
        assert choose_model(greeting) == "echo"


def test_short_prompt_routes_to_echo():
    assert choose_model("hi there") == "echo"
    assert choose_model("10 chars..") == "echo"


def test_technical_prompt_routes_to_granite():
    assert choose_model("how do pods work in openshift") == "granite"
    assert choose_model("explain kubernetes deployments") == "granite"
    assert choose_model("what is an ai gateway") == "granite"


def test_other_prompts_route_to_tinyllama():
    assert choose_model("write a poem about the ocean") == "tinyllama"
    assert choose_model(
        "what is the best recipe for pizza dough"
    ) == "tinyllama"


def test_whitespace_only_greeting_is_stripped():
    assert choose_model("  hello  ") == "echo"


def test_case_insensitive_greeting():
    assert choose_model("HELLO") == "echo"


def test_football_prompts_route_to_rag():
    assert choose_model("who won the champions league final") == "rag"
    assert choose_model("messi transfer news") == "rag"
    assert choose_model("premier league title race") == "rag"
    assert choose_model("what is the ballon d'or") == "rag"
    assert choose_model("best goalkeeper in la liga") == "rag"


def test_non_football_still_routes_to_tinyllama():
    assert choose_model("write a poem about the ocean") == "tinyllama"
    assert choose_model("what is the best recipe for pizza dough") == "tinyllama"
