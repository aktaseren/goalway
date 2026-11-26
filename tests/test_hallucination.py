"""Tests for gateway/rag/hallucination.py (groundedness checks)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.rag.hallucination import (
    check_groundedness,
    extract_claims,
    groundedness_report,
)
from gateway.rag.models import DocumentChunk, SearchResult


def make_result(doc_id: str, text: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=f"{doc_id}-0",
            document_id=doc_id,
            text=text,
            position=0,
        ),
        score=score,
    )


CONTEXT_A = (
    "The OpenShift AI Gateway serves models on Kubernetes pods. "
    "It supports per-key quotas and response caching."
)
CONTEXT_B = "GitOps deployments are managed with Argo CD."

GROUNDLESS_CLAIM = "The gateway runs on Mars and costs five dollars."


def test_extract_claims_splits_sentences():
    claims = extract_claims("First claim. Second claim! Third claim?")

    assert claims == [
        "First claim.",
        "Second claim!",
        "Third claim?",
    ]


def test_grounded_answer_is_grounded():
    answer = "The OpenShift AI Gateway serves models on Kubernetes pods."

    report = groundedness_report(
        answer, [make_result("doc-a", CONTEXT_A)]
    )

    assert report["verdict"] == "grounded"
    assert report["groundedness_score"] == 1.0
    assert report["supported_claims"] == 1
    assert report["total_claims"] == 1
    assert report["unsupported_claims"] == []


def test_fabricated_claim_is_detected():
    report = groundedness_report(
        GROUNDLESS_CLAIM, [make_result("doc-a", CONTEXT_A)]
    )

    assert report["verdict"] in ("likely_hallucinated", "partially_grounded")
    assert report["supported_claims"] == 0
    assert GROUNDLESS_CLAIM in report["unsupported_claims"]
    assert report["groundedness_score"] < 0.5


def test_fabricated_claim_flagged_in_check_groundedness():
    result = check_groundedness(GROUNDLESS_CLAIM, [CONTEXT_A])

    assert result["total_claims"] == 1
    assert result["supported_claims"] == 0
    assert result["claims"][0]["status"] == "unsupported"
    assert result["claims"][0]["support"] < 0.25


def test_partially_supported_answer():
    answer = (
        "The OpenShift AI Gateway serves models on Kubernetes pods. "
        + GROUNDLESS_CLAIM
    )

    report = groundedness_report(
        answer, [make_result("doc-a", CONTEXT_A)]
    )

    assert report["total_claims"] == 2
    assert report["supported_claims"] == 1
    assert report["groundedness_score"] == 0.5
    assert report["verdict"] == "partially_grounded"
    assert GROUNDLESS_CLAIM in report["unsupported_claims"]


def test_partially_supported_claim_status():
    # Claim whose content terms are half covered by the context.
    claim = "The gateway serves models on Kubernetes pods."

    result = check_groundedness(
        claim, ["The gateway serves quotas on worker nodes."],
        support_threshold=0.5,
    )

    assert result["claims"][0]["status"] in (
        "supported", "partially_supported",
    )


def test_empty_answer():
    report = groundedness_report("", [make_result("doc-a", CONTEXT_A)])

    assert report["total_claims"] == 0
    assert report["groundedness_score"] == 1.0
    assert report["verdict"] == "grounded"
    assert report["unsupported_claims"] == []


def test_supporting_documents_populated():
    answer = (
        "The OpenShift AI Gateway serves models on Kubernetes pods. "
        "GitOps deployments are managed with Argo CD."
    )
    results = [
        make_result("doc-a", CONTEXT_A),
        make_result("doc-b", CONTEXT_B),
    ]

    report = groundedness_report(answer, results)

    assert report["verdict"] == "grounded"
    assert report["supporting_documents"] == ["doc-a", "doc-b"]


def test_best_chunk_index_set():
    answer = "GitOps deployments are managed with Argo CD."

    report = groundedness_report(
        answer,
        [make_result("doc-a", CONTEXT_A), make_result("doc-b", CONTEXT_B)],
    )

    claims = report["claims"]
    assert claims[0]["best_chunk_index"] == 1


def test_summary_mentions_counts():
    report = groundedness_report(
        "The OpenShift AI Gateway serves models on Kubernetes pods.",
        [make_result("doc-a", CONTEXT_A)],
    )

    assert "1/1 claims supported" in report["summary"]
    assert "grounded" in report["summary"]
