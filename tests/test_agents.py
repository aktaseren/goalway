"""Tests for gateway/agents/orchestrator.py (multi-agent pipeline)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from types import SimpleNamespace

import pytest

import gateway.api as api_module
import gateway.agents.orchestrator as orchestrator_module
from gateway.agents import Orchestrator
from gateway.llm import LLMError
from gateway.rag.models import DocumentChunk, SearchResult

PLANNER_PLAN = '["Explain the quota policy", "Explain the caching policy"]'

RAG_CONTEXT = (
    "Quotas limit requests per API key. "
    "Caching stores responses for repeated requests."
)


class FakeRAGService:
    def search(self, query: str, top_k: int = 3, **kwargs):
        return (
            [
                SearchResult(
                    chunk=DocumentChunk(
                        chunk_id="doc-1-0",
                        document_id="doc-1",
                        text=RAG_CONTEXT,
                        position=0,
                    ),
                    score=0.9,
                )
            ],
            SimpleNamespace(strategy="hybrid"),
        )


def test_extract_json_strips_code_fences():
    extract = Orchestrator._extract_json

    assert extract('["a", "b"]') == '["a", "b"]'
    assert extract('```json\n["a", "b"]\n```') == '["a", "b"]'
    assert extract('```\n{"verdict": "approve"}\n```') == (
        '{"verdict": "approve"}'
    )
    assert extract('  {"a": 1}  ') == '{"a": 1}'


def role_complete_text(planner_reply, executor_reply, reviewer_reply):
    """
    Build a deterministic complete_text fake that answers based on
    the system prompt of the calling role.
    """
    def fake(model, prompt, system=None, temperature=None):
        if "planning agent" in system:
            return planner_reply
        if "research agent" in system:
            return "Research notes: " + RAG_CONTEXT
        if "executor agent" in system:
            return executor_reply(prompt)
        if "reviewer agent" in system:
            if isinstance(reviewer_reply, Exception):
                raise reviewer_reply
            return reviewer_reply
        raise AssertionError(f"unexpected system prompt: {system!r}")

    return fake


def test_full_pipeline_two_steps(monkeypatch):
    def executor_reply(prompt: str) -> str:
        if "quota" in prompt:
            return "Quotas limit requests per API key."
        return "Caching stores responses for repeated requests."

    monkeypatch.setattr(
        orchestrator_module,
        "complete_text",
        role_complete_text(PLANNER_PLAN, executor_reply,
                           '{"verdict": "approve", "feedback": "all good"}'),
    )
    monkeypatch.setattr(api_module, "rag_service", FakeRAGService())

    result = Orchestrator().run("Explain quotas and caching.")

    data = result.to_dict()

    assert data["run_id"].startswith("run-")
    assert data["objective"] == "Explain quotas and caching."
    assert data["status"] == "approve"

    # Planner produced two steps.
    assert len(data["steps"]) == 2
    assert data["steps"][0]["step"] == "Explain the quota policy"
    assert data["steps"][1]["step"] == "Explain the caching policy"

    for step in data["steps"]:
        assert "research_ms" in step
        assert "execute_ms" in step
        assert "documents_used" in step
        assert "answer" in step

    # Researcher found the fake document for each step.
    assert data["steps"][0]["documents_used"] == ["doc-1"]

    # Final answer combines both step answers.
    assert "Quotas limit requests per API key." in data["answer"]
    assert "Caching stores responses" in data["answer"]

    # Review verdict + groundedness report.
    assert data["review"]["verdict"] == "approve"
    assert data["review"]["feedback"] == "all good"
    assert data["review"]["groundedness"]["verdict"] == "grounded"
    assert data["review"]["groundedness"]["supported_claims"] == 2
    assert data["review"]["groundedness"]["total_claims"] == 2

    # Trace: planner, then researcher+executor per step, then reviewer.
    agents = [entry["agent"] for entry in data["trace"]]
    assert agents == [
        "planner",
        "researcher", "executor",
        "researcher", "executor",
        "reviewer",
    ]
    assert data["trace"][0]["steps_planned"] == 2
    assert data["trace"][-1]["verdict"] == "approve"

    assert isinstance(data["duration_ms"], float)


def test_planner_garbage_falls_back_to_single_step(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "complete_text",
        role_complete_text(
            "Sorry, I cannot plan that for you.",
            lambda prompt: "Some answer.",
            '{"verdict": "approve", "feedback": "ok"}',
        ),
    )
    monkeypatch.setattr(api_module, "rag_service", FakeRAGService())

    result = Orchestrator().run("Do everything.")

    data = result.to_dict()

    # Fallback plan: one step covering the whole objective.
    assert len(data["steps"]) == 1
    assert data["steps"][0]["step"] == "Do everything."
    assert data["answer"].startswith("**Step 1: Do everything.**")


def test_reviewer_model_failure_falls_back_to_deterministic(monkeypatch):
    # Executor produces an unsupported claim -> deterministic check
    # flags it, and the reviewer model is unavailable (LLMError).
    monkeypatch.setattr(
        orchestrator_module,
        "complete_text",
        role_complete_text(
            PLANNER_PLAN,
            lambda prompt: "The gateway is built entirely from cheese.",
            LLMError("reviewer backend down"),
        ),
    )
    monkeypatch.setattr(api_module, "rag_service", FakeRAGService())

    result = Orchestrator().run("Describe the gateway architecture.")

    review = result.to_dict()["review"]

    assert review["verdict"] == "revise"
    assert "not grounded" in review["feedback"]
    assert review["groundedness"]["verdict"] == "likely_hallucinated"
    assert review["groundedness"]["supported_claims"] == 0


def test_use_rag_false_skips_research(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "complete_text",
        role_complete_text(
            '["Only step"]',
            lambda prompt: "Answer without research.",
            '{"verdict": "approve", "feedback": "ok"}',
        ),
    )

    result = Orchestrator(use_rag=False).run("Objective.")

    data = result.to_dict()
    assert data["steps"][0]["documents_used"] == []
    # Groundedness check with no context -> unsupported claims.
    assert data["review"]["groundedness"]["total_claims"] == 1
