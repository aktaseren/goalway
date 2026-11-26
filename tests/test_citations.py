"""Tests for gateway/rag/citations.py (source spans + citations)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.rag.citations import (
    build_citations,
    find_source_spans,
    format_answer_with_citations,
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


CHUNK_TEXT = (
    "OpenShift pods run containers. "
    "Pods are scheduled onto worker nodes. "
    "The weather is lovely today."
)


def test_find_source_spans_matching_sentence():
    spans = find_source_spans("pods containers", CHUNK_TEXT)

    assert spans, "expected at least one matching span"
    assert spans[0]["text"] == "OpenShift pods run containers."
    # Char offsets must point at the matched sentence inside the chunk.
    assert (
        CHUNK_TEXT[spans[0]["start"]:spans[0]["end"]].strip()
        == spans[0]["text"]
    )
    assert spans[0]["score"] > 0


def test_find_source_spans_offsets_exact():
    spans = find_source_spans("pods containers", CHUNK_TEXT)

    for span in spans:
        assert 0 <= span["start"] < span["end"] <= len(CHUNK_TEXT)
        assert CHUNK_TEXT[span["start"]:span["end"]].strip() == span["text"]


def test_find_source_spans_unrelated_query():
    spans = find_source_spans("quantum entanglement physics", CHUNK_TEXT)

    assert spans == []


def test_find_source_spans_sorted_by_start():
    # Both first sentences mention pods -> both pass the threshold.
    spans = find_source_spans("pods", CHUNK_TEXT)

    assert len(spans) == 2
    starts = [span["start"] for span in spans]
    assert starts == sorted(starts)


def test_find_source_spans_max_spans_respected():
    spans = find_source_spans("pods", CHUNK_TEXT, max_spans=1)

    assert len(spans) == 1
    assert spans[0]["start"] == 0


def test_find_source_spans_empty_query():
    assert find_source_spans("", CHUNK_TEXT) == []
    assert find_source_spans("?!.", CHUNK_TEXT) == []


def test_build_citations_numbering_and_fields():
    results = [
        make_result("doc-1", CHUNK_TEXT, score=0.9),
        make_result("doc-2", "Pods are great.", score=0.4),
    ]

    citations = build_citations(results, "pods containers")

    assert [c["citation_id"] for c in citations] == ["[1]", "[2]"]
    assert citations[0]["document_id"] == "doc-1"
    assert citations[0]["chunk_id"] == "doc-1-0"
    assert citations[0]["position"] == 0
    assert citations[0]["relevance_score"] == 0.9
    assert isinstance(citations[0]["spans"], list)


def test_build_citations_empty_results():
    assert build_citations([], "pods") == []


def test_format_answer_with_citations_appends_sources():
    citations = build_citations(
        [make_result("doc-1", CHUNK_TEXT)],
        "pods containers",
    )

    formatted = format_answer_with_citations("Pods run containers.", citations)

    assert formatted.startswith("Pods run containers.")
    assert "Sources:" in formatted
    assert "[1] doc-1" in formatted
    # The first matching span text is used as a preview.
    assert "OpenShift pods run containers" in formatted


def test_format_answer_with_citations_no_citations():
    assert format_answer_with_citations("Plain answer.", []) == "Plain answer."


def test_format_answer_with_citations_multiple_sources():
    citations = [
        {"citation_id": "[1]", "document_id": "doc-a", "spans": []},
        {"citation_id": "[2]", "document_id": "doc-b", "spans": []},
    ]

    formatted = format_answer_with_citations("Answer.", citations)

    assert "[1] doc-a" in formatted
    assert "[2] doc-b" in formatted
