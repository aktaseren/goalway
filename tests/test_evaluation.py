"""Tests for gateway/rag/evaluation.py (retrieval metrics + judge)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math

import pytest

import gateway.llm
from gateway.rag.evaluation import (
    _parse_judge_response,
    evaluate_retrieval,
    llm_judge,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from gateway.rag.models import DocumentChunk, SearchResult


def make_result(doc_id: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=f"{doc_id}-0",
            document_id=doc_id,
            text=f"text of {doc_id}",
            position=0,
        ),
        score=score,
    )


def ranked(*doc_ids: str) -> list[SearchResult]:
    return [make_result(doc_id) for doc_id in doc_ids]


# Ranking: d1, d2, d3, d4. Relevant: d2, d4. k = 3.
RESULTS = ranked("d1", "d2", "d3", "d4")
RELEVANT = {"d2", "d4"}


def test_recall_at_k_hand_computed():
    # top-3 = {d1, d2, d3}, relevant found: {d2} -> 1/2
    assert recall_at_k(RESULTS, RELEVANT, k=3) == 0.5


def test_recall_at_k_no_relevant():
    assert recall_at_k(RESULTS, set(), k=3) == 0.0


def test_precision_at_k_hand_computed():
    # top-3 = {d1, d2, d3}, 1 of 3 relevant -> 1/3
    assert precision_at_k(RESULTS, RELEVANT, k=3) == pytest.approx(1 / 3)


def test_precision_at_k_empty_results():
    assert precision_at_k([], RELEVANT, k=3) == 0.0


def test_mrr_hand_computed():
    # First relevant result (d2) is at rank 2 -> 1/2
    assert mrr(RESULTS, RELEVANT) == 0.5


def test_mrr_no_relevant():
    assert mrr(RESULTS, {"missing"}) == 0.0


def test_ndcg_at_k_hand_computed():
    # dcg = 1/log2(3); idcg = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RESULTS, RELEVANT, k=3) == pytest.approx(expected)


def test_perfect_ranking_all_ones():
    results = ranked("d2", "d4", "d1")

    assert recall_at_k(results, RELEVANT, k=2) == 1.0
    assert precision_at_k(results, RELEVANT, k=2) == 1.0
    assert mrr(results, RELEVANT) == 1.0
    assert ndcg_at_k(results, RELEVANT, k=2) == 1.0


def test_evaluate_retrieval_shape():
    report = evaluate_retrieval("pods", RESULTS, ["d2", "d4"], k=3)

    assert report["query"] == "pods"
    assert report["k"] == 3
    assert report["returned_results"] == 4
    assert report["relevant_ids"] == ["d2", "d4"]
    assert report["recall_at_k"] == 0.5
    assert report["mrr"] == 0.5
    assert report["precision_at_k"] == round(1 / 3, 4)
    assert report["ndcg_at_k"] > 0


def test_parse_judge_response_plain_json():
    parsed = _parse_judge_response(
        '{"score": 4, "grounded": true, "reasoning": "looks fine"}'
    )

    assert parsed == {
        "score": 4,
        "grounded": True,
        "reasoning": "looks fine",
    }


def test_parse_judge_response_code_fences():
    parsed = _parse_judge_response(
        '```json\n{"score": 5, "grounded": true, "reasoning": "great"}\n```'
    )

    assert parsed is not None
    assert parsed["score"] == 5
    assert parsed["grounded"] is True


def test_parse_judge_response_with_prose():
    parsed = _parse_judge_response(
        'Here is my evaluation:\n{"score": 2, "grounded": false}\nthanks'
    )

    assert parsed is not None
    assert parsed["score"] == 2
    assert parsed["grounded"] is False


def test_parse_judge_response_garbage():
    assert _parse_judge_response("no json here at all") is None
    assert _parse_judge_response("{broken json") is None
    assert _parse_judge_response('{"grounded": true}') is None  # no score


def test_parse_judge_response_clamps_score():
    parsed = _parse_judge_response('{"score": 99, "grounded": true}')
    assert parsed["score"] == 5

    parsed = _parse_judge_response('{"score": -3, "grounded": false}')
    assert parsed["score"] == 1


def _judge_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_llm_judge_uses_model(monkeypatch):
    monkeypatch.setattr(
        gateway.llm,
        "complete",
        lambda **kwargs: _judge_response(
            '{"score": 4, "grounded": true, "reasoning": "supported"}'
        ),
    )

    verdict = llm_judge("query", "answer", [make_result("d1")], model="auto")

    assert verdict["judge"] == "llm"
    assert verdict["model"] == "auto"
    assert verdict["score"] == 4
    assert verdict["grounded"] is True
    assert verdict["reasoning"] == "supported"


def test_llm_judge_falls_back_to_heuristic_on_error(monkeypatch):
    def raise_error(**kwargs):
        raise RuntimeError("no model reachable")

    monkeypatch.setattr(gateway.llm, "complete", raise_error)

    context = "The OpenShift AI Gateway serves models on Kubernetes pods."
    results = [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="d1-0",
                document_id="d1",
                text=context,
                position=0,
            ),
            score=0.9,
        )
    ]

    verdict = llm_judge(
        "what does the gateway serve",
        "The OpenShift AI Gateway serves models on Kubernetes pods.",
        results,
    )

    assert verdict["judge"] == "heuristic"
    assert verdict["model"] is None
    assert 1 <= verdict["score"] <= 5
    assert verdict["grounded"] is True
    assert verdict["reasoning"]


def test_llm_judge_accepts_openai_shaped_answer(monkeypatch):
    monkeypatch.setattr(
        gateway.llm,
        "complete",
        lambda **kwargs: _judge_response(
            '{"score": 3, "grounded": false, "reasoning": "partial"}'
        ),
    )

    answer = {"choices": [{"message": {"content": "some answer"}}]}

    verdict = llm_judge("query", answer, [make_result("d1")])

    assert verdict["judge"] == "llm"
    assert verdict["score"] == 3
