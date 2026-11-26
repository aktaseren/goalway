"""
Hallucination detection via groundedness checking .

An answer is split into atomic claims (approximated as sentences).
Each claim is checked for support in the retrieved context using
weighted token overlap. Claims below the support threshold are
flagged as unsupported, and an overall groundedness score plus
verdict is produced.

This is a cheap, deterministic first line of defense; the
LLM-as-judge in gateway.rag.evaluation gives a deeper, model-based
assessment.
"""

import re

from gateway.rag.citations import _tokenize

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
PARTIALLY_SUPPORTED = "partially_supported"


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _claim_support(
    claim: str,
    contexts: list[str],
    term_weights: dict[str, float] | None = None,
) -> float:
    """
    Support score for a claim against one context chunk.

    Uses weighted coverage of the claim's content terms: terms are
    weighted by inverse document frequency across the retrieved
    contexts, so distinctive terms (names, places, numbers) dominate
    the score while common words contribute little. With a single
    context this reduces to plain coverage.
    """
    claim_terms = [t for t in _tokenize(claim) if len(t) > 2]
    context_terms = set(_tokenize(contexts[0]))

    if not claim_terms:
        return 1.0

    stop_words = _STOP_WORDS

    content_terms = [t for t in claim_terms if t not in stop_words]

    if not content_terms:
        return 1.0

    if term_weights is None:
        matched = sum(1 for t in content_terms if t in context_terms)
        return matched / len(content_terms)

    matched_weight = 0.0
    total_weight = 0.0

    for term in content_terms:
        weight = term_weights.get(term, 1.0)
        total_weight += weight

        if term in context_terms:
            matched_weight += weight

    return matched_weight / total_weight if total_weight > 0 else 1.0


_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any",
    "can", "had", "her", "was", "one", "our", "out", "day", "get",
    "has", "him", "his", "how", "man", "new", "now", "old", "see",
    "two", "way", "who", "did", "its", "let", "put", "say", "she",
    "too", "use", "that", "with", "have", "this", "will", "your",
    "from", "they", "know", "want", "been", "good", "much", "some",
    "time", "very", "when", "come", "here", "just", "like", "long",
    "make", "many", "over", "such", "take", "than", "them", "well",
    "what", "were", "would", "could", "should", "into", "also",
    "based", "using",
}


def _term_weights(contexts: list[str]) -> dict[str, float]:
    """
    Inverse document frequency of every claim-relevant term across
    the retrieved contexts. Terms absent everywhere keep weight 1.0
    (maximum penalty when unmatched).
    """
    context_term_sets = [set(_tokenize(context)) for context in contexts]

    vocabulary: set[str] = set()

    for term_set in context_term_sets:
        vocabulary |= term_set

    weights: dict[str, float] = {}

    for term in vocabulary:
        df = sum(1 for term_set in context_term_sets if term in term_set)
        weights[term] = 1.0 / max(df, 1)

    return weights


def extract_claims(answer: str) -> list[str]:
    """Split an answer into atomic claims (sentences)."""
    return _sentences(answer)


def check_groundedness(
    answer: str,
    contexts: list[str],
    support_threshold: float = 0.5,
) -> dict:
    """
    Check each claim of the answer against the combined context.

    Returns:
    {
        "claims": [{"claim", "support", "status", "best_chunk_index"}],
        "groundedness_score": 0..1,
        "supported_claims": int,
        "total_claims": int,
    }
    """
    claims = extract_claims(answer)
    results = []
    supported = 0

    weights = _term_weights(contexts)

    for claim in claims:
        best_support = 0.0
        best_index = -1

        for index, context in enumerate(contexts):
            support = _claim_support(claim, [context], term_weights=weights)

            if support > best_support:
                best_support = support
                best_index = index

        if best_support >= support_threshold:
            status = SUPPORTED
            supported += 1
        elif best_support >= support_threshold / 2:
            status = PARTIALLY_SUPPORTED
        else:
            status = UNSUPPORTED

        results.append(
            {
                "claim": claim,
                "support": round(best_support, 4),
                "status": status,
                "best_chunk_index": best_index,
            }
        )

    return {
        "claims": results,
        "groundedness_score": round(supported / len(claims), 4) if claims else 1.0,
        "supported_claims": supported,
        "total_claims": len(claims),
    }


def groundedness_report(answer: str, results: list) -> dict:
    """
    Full hallucination report for an answer given SearchResult-like
    objects. Produces a verdict and flags hallucinated claims.
    Duplicate chunks (e.g. the same document retrieved for two agent
    steps) are checked once.
    """
    seen_chunk_ids: set[str] = set()
    unique_results = []

    for result in results:
        chunk_id = getattr(result.chunk, "chunk_id", None)

        if chunk_id is not None and chunk_id in seen_chunk_ids:
            continue

        if chunk_id is not None:
            seen_chunk_ids.add(chunk_id)

        unique_results.append(result)

    contexts = [result.chunk.text for result in unique_results]

    report = check_groundedness(answer, contexts)

    unsupported = [
        claim
        for claim in report["claims"]
        if claim["status"] == UNSUPPORTED
    ]

    score = report["groundedness_score"]

    if score >= 0.8:
        verdict = "grounded"
    elif score >= 0.5:
        verdict = "partially_grounded"
    else:
        verdict = "likely_hallucinated"

    # Reference the documents that support the answer.
    supported_doc_ids = sorted(
        {
            unique_results[claim["best_chunk_index"]].chunk.document_id
            for claim in report["claims"]
            if claim["best_chunk_index"] >= 0
            and claim["best_chunk_index"] < len(unique_results)
        }
    )

    return {
        "verdict": verdict,
        "groundedness_score": score,
        "supported_claims": report["supported_claims"],
        "total_claims": report["total_claims"],
        "unsupported_claims": [
            claim["claim"] for claim in unsupported
        ],
        "supporting_documents": supported_doc_ids,
        "claims": report["claims"],
        "summary": (
            f"{report['supported_claims']}/{report['total_claims']} claims "
            f"supported by retrieved context ({verdict})"
        ),
    }
