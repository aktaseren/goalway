"""
Extractive question answering over the RAG knowledge base .

Powers the built-in "rag" chat model: a question is answered by
retrieving the most relevant ingested documents and extracting the
sentences that best match the query -- deterministic, dependency-free
and honest about its sources (every answer carries citations).

When real LLM backends (granite, vLLM, ...) are deployed, prompts are
augmented with the same retrieval and the model generates the answer;
this extractive path guarantees the demo works with zero model
infrastructure.
"""

from gateway.rag.citations import build_citations, _tokenize

# Content words that carry no topical signal for coverage gating.
_GATE_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any",
    "can", "had", "her", "was", "one", "our", "out", "get", "has",
    "him", "his", "how", "man", "new", "now", "old", "see", "two",
    "way", "who", "did", "its", "let", "put", "say", "she", "too",
    "use", "that", "with", "have", "this", "will", "your", "from",
    "they", "know", "want", "been", "good", "much", "some", "time",
    "very", "when", "come", "here", "just", "like", "long", "make",
    "many", "over", "such", "take", "than", "them", "well", "what",
    "were", "would", "could", "should", "into", "also", "about",
    "tell", "me", "him", "her", "there", "then", "them", "is",
    "are", "was", "were", "am", "do", "does", "did", "doing",
}


def _stem(token: str) -> str:
    """
    Naive singularization: strip one trailing 's'. Good enough to
    match champion/champions, name/names, goal/goals — irregular
    verbs (won/winning) are out of scope.
    """
    if token.endswith("s") and len(token) > 3:
        return token[:-1]

    return token


def content_coverage(query: str, text: str) -> float:
    """
    Fraction of the query's content terms (stop words stripped,
    lightly stemmed) present in the text. Separates topical matches
    (0.5-1.0) from stop-word-only coincidences (~0.0) far better than
    raw lexical scores, which common words inflate.

    Used as the relevance gate for extractive answers: below the
    threshold the model says "not in my knowledge base" instead of
    quoting marginally-related text.
    """
    query_terms = [
        _stem(token)
        for token in _tokenize(query)
        if len(token) > 2 and token not in _GATE_STOP_WORDS
    ]

    if not query_terms:
        return 0.0

    text_terms = {_stem(token) for token in _tokenize(text)}

    matched = sum(1 for term in query_terms if term in text_terms)

    return matched / len(query_terms)


def _recency_order(results: list) -> list:
    """
    News-aware ordering: among results whose relevance scores are
    close (within 2x of the best), put the most recent first based
    on the chunk's year metadata. Clearly-more-relevant results keep
    their lead.
    """
    if not results:
        return results

    scored = [
        (
            result.rerank_score
            if result.rerank_score is not None
            else result.score,
            result,
        )
        for result in results
    ]

    best = max(score for score, _ in scored)

    close: list = []
    rest: list = []

    for score, result in scored:
        if score >= best * 0.5:
            close.append(result)
        else:
            rest.append(result)

    close.sort(
        key=lambda result: result.chunk.metadata.get("year") or 0,
        reverse=True,
    )

    return close + rest


# Questions that expect a list, not a single sentence.
LIST_INTENT_TERMS = [
    "names",
    "name",
    "members",
    "member",
    "list",
    "squad",
    "roster",
    "lineup",
    "players",
    "all of",
    "who is in",
    "who are",
]


def is_list_question(query: str) -> bool:
    """Does the question ask for a list of things (names, members...)?"""
    query_lower = query.lower()

    return any(term in query_lower for term in LIST_INTENT_TERMS)


# Queries whose intent is the most RECENT result, not just the most
# relevant one.
RECENCY_INTENT_TERMS = [
    "last",
    "latest",
    "recent",
    "most recent",
    "newest",
    "current",
    "currently",
    "this year",
    "today",
]


def has_recency_intent(query: str) -> bool:
    """Does the query ask for the latest thing ('last champion...')?

    Lexical rankers have no notion of time — 'last' is just another
    unmatched token. Queries with recency intent should be ordered
    newest-first among close-scoring results.
    """
    query_lower = query.lower()

    return any(term in query_lower for term in RECENCY_INTENT_TERMS)


def build_extractive_answer(
    query: str,
    results: list,
    max_sentences: int = 3,
    fetch_document=None,
) -> tuple[str, list[dict]]:
    """
    Build an answer from retrieved results.

    Returns (answer_text, citations). The answer quotes the single
    most query-relevant sentence of each of the top results, marked
    with its citation id, followed by a Sources block. List-type
    questions (names, members, squad...) are answered with the full
    text of the single best-matching document — all of its chunks,
    fetched via fetch_document when available.
    """
    if not results:
        return (
            "I don't have anything about that in the knowledge base. "
            "Ingest documents first (RAG tab → Load sample football news).",
            [],
        )

    list_mode = is_list_question(query)

    # Order first, then build citations from the ordered results so
    # citation ids stay aligned with the quoted documents.
    ordered = _recency_order(results)

    if list_mode:
        # One full document answers a list; more would bury it.
        # Select by content coverage of the question rather than
        # lexical rank — ranking keys on term frequency, so a short
        # doc repeating the club name can outrank the actual list.
        ordered = sorted(
            ordered,
            key=lambda result: content_coverage(
                query, result.chunk.text
            ),
            reverse=True,
        )[:1]
    else:
        ordered = ordered[:max_sentences]

    citations = build_citations(ordered, query)

    parts = []

    for citation, result in zip(citations, ordered):
        spans = citation.get("spans") or []

        if list_mode:
            # Quote the WHOLE document (all chunks, in order) — a
            # list spans the entire text, and the chunker may have
            # split it across several chunks.
            sentence = None

            if fetch_document is not None:
                chunks = fetch_document(result.chunk.document_id)

                if chunks:
                    sentence = " ".join(
                        chunk.text.strip() for chunk in chunks
                    )

            if sentence is None:
                sentence = result.chunk.text
        elif spans:
            best_span = max(spans, key=lambda span: span["score"])
            sentence = best_span["text"]
        else:
            # No query-relevant sentence: fall back to the first
            # sentence of the chunk.
            sentence = result.chunk.text.split(". ")[0].rstrip(".") + "."

        parts.append(f"{sentence} {citation['citation_id']}")

    answer = " ".join(parts)

    answer += "\n\nSources: " + ", ".join(
        f"{c['citation_id']} {c['document_id']}"
        for c in citations
    )

    return answer, citations
