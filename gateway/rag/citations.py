"""
Citations with source spans .

Every search result can be turned into a citation that points at the
exact span(s) inside the source chunk that matched the query. The
span finder is deliberately simple and dependency-free: it scores the
chunk's sentences by query-term overlap and returns the best matching
sentences with character offsets, so a client can highlight them.
"""

import re

from gateway.rag.models import DocumentChunk


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", text)
    ]


def _sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentences as (sentence, start, end) where
    text[start:end] == sentence exactly (no surrounding whitespace).
    """
    results = []
    start = 0

    for match in re.finditer(r"[.!?]+[\s]+|\n+|[.!?]+$", text):
        end = match.end()

        if text[start:end].strip():
            results.append(_exact_span(text, start, end))

        start = end

    if start < len(text):
        if text[start:].strip():
            results.append(_exact_span(text, start, len(text)))

    return results


def _exact_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    """
    Trim whitespace from both ends of text[start:end], returning the
    stripped sentence with offsets that satisfy
    text[new_start:new_end] == sentence.
    """
    raw = text[start:end]

    new_start = start + (len(raw) - len(raw.lstrip()))
    new_end = new_start + len(raw.strip())

    return raw.strip(), new_start, new_end


def _overlap_score(query_terms: set[str], sentence: str) -> float:
    if not query_terms:
        return 0.0

    sentence_terms = set(_tokenize(sentence))

    if not sentence_terms:
        return 0.0

    return len(query_terms & sentence_terms) / len(query_terms)


def find_source_spans(
    query: str,
    chunk_text: str,
    max_spans: int = 3,
    min_score: float = 0.2,
) -> list[dict]:
    """
    Find the spans in chunk_text most relevant to the query.

    Returns a list of:
    {
        "text": "<matched sentence>",
        "start": <char offset>,
        "end": <char offset (exclusive)>,
        "score": <0..1 query term overlap>
    }
    """
    query_terms = set(_tokenize(query))

    if not query_terms:
        return []

    scored = []

    for sentence, start, end in _sentences_with_offsets(chunk_text):
        score = _overlap_score(query_terms, sentence)

        if score >= min_score:
            scored.append(
                {
                    "text": sentence,
                    "start": start,
                    "end": end,
                    "score": round(score, 4),
                }
            )

    scored.sort(key=lambda span: span["score"], reverse=True)

    spans = scored[:max_spans]
    spans.sort(key=lambda span: span["start"])

    return spans


def build_citation(
    index: int,
    chunk: DocumentChunk,
    score: float,
    query: str,
    max_spans: int = 3,
) -> dict:
    """
    Build a citation object for a search result.

    Shape:
    {
        "citation_id": "[1]",
        "document_id": ...,
        "chunk_id": ...,
        "position": ...,
        "relevance_score": ...,
        "spans": [ {text, start, end, score}, ... ]
    }
    """
    return {
        "citation_id": f"[{index}]",
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "position": chunk.position,
        "relevance_score": round(score, 4),
        "spans": find_source_spans(
            query,
            chunk.text,
            max_spans=max_spans,
        ),
    }


def build_citations(
    results: list,
    query: str,
    max_spans: int = 3,
) -> list[dict]:
    """Build citations for an ordered list of SearchResult objects."""
    citations = []

    for index, result in enumerate(results, start=1):
        citations.append(
            build_citation(
                index,
                result.chunk,
                result.score,
                query,
                max_spans=max_spans,
            )
        )

    return citations


def format_answer_with_citations(
    answer: str,
    citations: list[dict],
) -> str:
    """
    Append a 'Sources' block to an answer, referencing citations by
    their citation_id and document_id.
    """
    if not citations:
        return answer

    lines = [answer, "", "Sources:"]

    for citation in citations:
        preview = ""

        if citation.get("spans"):
            preview = citation["spans"][0]["text"][:120]
        else:
            preview = ""

        lines.append(
            f"{citation['citation_id']} {citation['document_id']}"
            f"{': ' + preview if preview else ''}"
        )

    return "\n".join(lines)
