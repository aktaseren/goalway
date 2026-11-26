"""Tests for gateway/rag/rerankers/ (second-stage reranking)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import requests

import gateway.rag.rerankers.cross_encoder as cross_encoder_module
import gateway.rag.rerankers.factory as factory_module
from gateway.rag.models import DocumentChunk, SearchResult
from gateway.rag.rerankers import (
    CrossEncoderReranker,
    LexicalReranker,
    create_reranker,
)


def make_result(doc_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=f"{doc_id}-0",
            document_id=doc_id,
            text=text,
            position=0,
        ),
        score=score,
    )


RELEVANT_TEXT = (
    "OpenShift pod scheduling places pods onto worker nodes "
    "based on resource requests."
)
IRRELEVANT_TEXT = "Cooking pasta requires boiling water and salt."


def test_lexical_reranker_reorders_results():
    # Relevant chunk has a much worse first-stage score.
    results = [
        make_result("irrelevant", IRRELEVANT_TEXT, 0.9),
        make_result("relevant", RELEVANT_TEXT, 0.05),
    ]

    reranker = LexicalReranker()
    reranked = reranker.rerank("openshift pod scheduling nodes", results)

    assert reranked[0].chunk.document_id == "relevant"
    assert reranked[1].chunk.document_id == "irrelevant"


def test_lexical_reranker_preserves_fields_and_sets_rerank_metadata():
    results = [make_result("relevant", RELEVANT_TEXT, 0.05)]

    reranked = LexicalReranker().rerank(
        "openshift pod scheduling", results
    )

    assert len(reranked) == 1
    result = reranked[0]

    # Original fields survive the model_copy.
    assert result.chunk.document_id == "relevant"
    assert result.chunk.text == RELEVANT_TEXT
    assert result.chunk.chunk_id == "relevant-0"
    assert result.score == 0.05

    # rerank metadata is attached.
    assert result.reranker == "lexical"
    assert result.rerank_score is not None
    assert result.rerank_score > 0


def test_lexical_reranker_respects_top_k():
    results = [make_result(f"doc-{i}", RELEVANT_TEXT, 0.1) for i in range(5)]

    reranked = LexicalReranker().rerank("pod scheduling", results, top_k=2)

    assert len(reranked) == 2


def test_lexical_reranker_empty_inputs():
    reranker = LexicalReranker()

    assert reranker.rerank("pods", []) == []
    # No query terms -> passthrough of the top_k slice.
    results = [make_result("doc-1", "text", 0.5)]
    assert reranker.rerank("?!.", results) == results


def test_lexical_reranker_name():
    assert LexicalReranker().name() == "lexical"


def test_relevant_outranks_irrelevant_despite_first_stage_score():
    results = [
        make_result("irrelevant", IRRELEVANT_TEXT, 0.99),
        make_result("relevant", RELEVANT_TEXT, 0.01),
    ]

    reranked = LexicalReranker(first_stage_weight=0.0).rerank(
        "openshift pod scheduling", results
    )

    assert reranked[0].chunk.document_id == "relevant"


def test_factory_returns_lexical_by_default(monkeypatch):
    monkeypatch.setattr(factory_module, "RAG_RERANKER", "lexical")

    reranker = create_reranker()

    assert isinstance(reranker, LexicalReranker)
    assert reranker.name() == "lexical"


def test_factory_explicit_lexical():
    assert isinstance(create_reranker("lexical"), LexicalReranker)


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError):
        create_reranker("does_not_exist")


def test_factory_cross_encoder_without_url_raises(monkeypatch):
    monkeypatch.setattr(factory_module, "CROSS_ENCODER_URL", "")

    with pytest.raises(ValueError):
        create_reranker("cross_encoder")


def test_cross_encoder_falls_back_to_lexical_on_request_error(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("backend down")

    monkeypatch.setattr(
        cross_encoder_module.requests, "post", raise_connection_error
    )

    results = [
        make_result("irrelevant", IRRELEVANT_TEXT, 0.9),
        make_result("relevant", RELEVANT_TEXT, 0.05),
    ]

    reranker = CrossEncoderReranker("http://reranker.test")
    reranked = reranker.rerank("openshift pod scheduling", results)

    # Lexical fallback produced the ordering + metadata.
    assert reranked[0].chunk.document_id == "relevant"
    assert all(r.reranker == "lexical" for r in reranked)
    assert reranker.name() == "cross_encoder"


def test_cross_encoder_uses_remote_scores(monkeypatch):
    ranking = [
        {"index": 1, "score": 0.95},
        {"index": 0, "score": 0.10},
    ]

    fake_response = type(
        "FakeResponse",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: ranking,
        },
    )()

    monkeypatch.setattr(
        cross_encoder_module.requests,
        "post",
        lambda *args, **kwargs: fake_response,
    )

    results = [
        make_result("doc-a", "first", 0.9),
        make_result("doc-b", "second", 0.1),
    ]

    reranked = CrossEncoderReranker("http://reranker.test").rerank(
        "anything", results
    )

    assert [r.chunk.document_id for r in reranked] == ["doc-b", "doc-a"]
    assert reranked[0].reranker == "cross_encoder"
    assert reranked[0].rerank_score == 0.95
