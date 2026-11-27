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

import re

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



# Words that shape an enumeration request but carry no topical
# signal — headlines don't contain the words "news" or "sport".
ENUMERATION_TERMS = {
    "top",
    "news",
    "headline",
    "headlines",
    "story",
    "stories",
    "article",
    "articles",
    "item",
    "items",
    "today",
    "todays",
    "sport",
    "sports",
    "show",
    "give",
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


def _gate_terms(
    query: str,
    extra_stop: frozenset | set = frozenset(),
) -> list[str]:
    """Content terms used for relevance gating, stemmed."""
    return [
        _stem(token)
        for token in _tokenize(query)
        if len(token) > 2
        and token not in _GATE_STOP_WORDS
        and token not in extra_stop
    ]


def _coverage_of_terms(terms: list[str], text: str) -> float | None:
    """Coverage of prepared terms against text; None when there are
    no terms to gate on (query is pure shape, e.g. 'top 5 news')."""
    if not terms:
        return None

    text_terms = {_stem(token) for token in _tokenize(text)}

    return sum(1 for term in terms if term in text_terms) / len(terms)


def content_coverage(
    query: str,
    text: str,
    extra_stop: frozenset | set = frozenset(),
) -> float:
    """
    Fraction of the query's content terms (stop words stripped,
    lightly stemmed) present in the text. Separates topical matches
    (0.5-1.0) from stop-word-only coincidences (~0.0) far better than
    raw lexical scores, which common words inflate.

    Used as the relevance gate for extractive answers: below the
    threshold the model says "not in my knowledge base" instead of
    quoting marginally-related text.
    """
    terms = _gate_terms(query, extra_stop)

    if not terms:
        return 0.0

    coverage = _coverage_of_terms(terms, text)

    return coverage if coverage is not None else 0.0


def gate_coverage(
    query: str,
    text: str,
    extra_stop: frozenset | set = frozenset(),
) -> float | None:
    """
    Coverage for gating, or None when the query carries no topical
    terms at all (pure enumeration shape) — in which case the gate
    should be skipped rather than declined.
    """
    return _coverage_of_terms(_gate_terms(query, extra_stop), text)


# Generic words that never name a topic ("is there any news for
# them?" -> the topic is whatever 'them' refers to, carried by the
# entity words around it).
TOPIC_STOP_WORDS = _GATE_STOP_WORDS | ENUMERATION_TERMS | {
    "support",
    "them",
    "their",
    "they",
    "team",
    "teams",
    "club",
    "clubs",
    "date",
    "time",
    "tell",
    "ask",
    "know",
    "want",
    "any",
    "something",
    "anything",
    # recency shape words, not entities
    "current",
    "currently",
    "latest",
    "last",
    "recent",
    "newest",
    "today",
    "todays",
}


def extract_topic(query: str) -> str | None:
    """
    The entity a question is about, for decline messages:
    'I support galatasaray. is there any news for them?' ->
    'galatasaray'. None when the query names nothing specific.
    """
    topic: list[str] = []
    seen: set[str] = set()

    for token in _tokenize(query):
        if (
            len(token) > 2
            and token not in TOPIC_STOP_WORDS
            and token not in seen
        ):
            seen.add(token)
            topic.append(token)

    # A short topic reads as an entity; a long one is a sentence —
    # fall back to the generic decline.
    if not topic or len(topic) > 2:
        return None

    return " ".join(topic)


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


# Enumeration questions: "top 5 news", "latest 3 headlines".
COUNT_PATTERNS = [
    re.compile(r"\btop\s+(\d{1,2})\b"),
    re.compile(r"\blatest\s+(\d{1,2})\b"),
    re.compile(r"\bfirst\s+(\d{1,2})\b"),
    re.compile(
        r"\b(\d{1,2})\s+(?:news|headlines?|stories|articles?|"
        r"items?|results?)\b"
    ),
]

# Words that shape an enumeration request but carry no topical
# signal — headlines don't contain the words "news" or "sport".
def extract_requested_count(query: str) -> int | None:
    """
    How many items does the query ask for? ("top 5 news" -> 5)

    Only structured count phrases match — a bare number like the
    year in "who won the 2023 final" is not a count.
    """
    for pattern in COUNT_PATTERNS:
        match = pattern.search(query.lower())

        if match:
            count = int(match.group(1))

            if 1 <= count <= 20:
                return count

    return None


def missing_recency_years(horizon: int | None) -> list[int]:
    """
    Years with no coverage between the KB horizon and today.

    For an annual event, the latest expected year is the current
    year from June onward (finals/titles are decided by then), the
    previous year before that. An empty list means the knowledge
    base is current.
    """
    if horizon is None:
        return []

    from datetime import date

    today = date.today()

    expected = today.year if today.month >= 6 else today.year - 1

    if horizon >= expected:
        return []

    return list(range(horizon + 1, expected + 1))


def recency_warning(horizon: int | None) -> str | None:
    """
    'No info found for 2026' style warning when the knowledge base
    is missing years up to the present. None when it is current.
    """
    missing = missing_recency_years(horizon)

    if not missing:
        return None

    if len(missing) == 1:
        years = str(missing[0])
    else:
        years = f"{missing[0]}-{missing[-1]}"

    return (
        f"No info found for {years}"
        + (
            f" — the latest information in the knowledge base is "
            f"from {horizon}"
            if horizon
            else ""
        )
    )


def build_extractive_answer(
    query: str,
    results: list,
    max_sentences: int = 3,
    fetch_document=None,
    requested_count: int | None = None,
) -> tuple[str, list[dict]]:
    """
    Build an answer from retrieved results.

    Returns (answer_text, citations). Three question shapes:

    - single-answer: quotes the most query-relevant sentence of each
      of the top results, marked with citation ids
    - list questions (names, members, squad...): the full text of
      the single best-matching document — all of its chunks, fetched
      via fetch_document when available
    - enumeration questions ("top 5 news"): N numbered items, newest
      first, one per document
    """
    if not results:
        return (
            "I don't have anything about that in the knowledge base. "
            "Ingest documents first (RAG tab → Load sample football news).",
            [],
        )

    list_mode = is_list_question(query) and not requested_count
    count_mode = requested_count is not None

    # Order first, then build citations from the ordered results so
    # citation ids stay aligned with the quoted documents.
    ordered = _recency_order(results)

    if count_mode:
        # Enumerations are newest-first news semantics.
        ordered = ordered[: requested_count]
    elif list_mode:
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
    separator = "\n" if count_mode else " "

    for index, (citation, result) in enumerate(
        zip(citations, ordered), start=1
    ):
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
        elif count_mode:
            # One item per document: the whole (short) item for
            # news, or its best span for longer documents.
            if len(result.chunk.text) <= 400:
                sentence = result.chunk.text
            elif spans:
                sentence = max(spans, key=lambda span: span["score"])[
                    "text"
                ]
            else:
                sentence = (
                    result.chunk.text.split(". ")[0].rstrip(".") + "."
                )
        elif spans:
            best_span = max(spans, key=lambda span: span["score"])
            sentence = best_span["text"]
        else:
            # No query-relevant sentence: fall back to the first
            # sentence of the chunk.
            sentence = result.chunk.text.split(". ")[0].rstrip(".") + "."

        prefix = f"{index}. " if count_mode else ""

        parts.append(f"{prefix}{sentence} {citation['citation_id']}")

    answer = separator.join(parts)

    answer += "\n\nSources: " + ", ".join(
        f"{c['citation_id']} {c['document_id']}"
        for c in citations
    )

    return answer, citations
