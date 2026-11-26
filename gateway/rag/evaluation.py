"""
RAG evaluation harness .

Two families of metrics:

1. Retrieval metrics (deterministic, no model needed):
   recall@k, precision@k, MRR and nDCG given a set of relevant
   document_ids for a query.

2. LLM-as-judge (groundedness / answer quality):
   scores an answer against retrieved context on a 1-5 scale. Uses
   the gateway's own model backends via gateway.llm.complete, with a
   deterministic lexical fallback when no model is reachable so the
   evaluation pipeline never hard-fails.
"""

import logging
import math
import re

from gateway.rag.hallucination import groundedness_report

logger = logging.getLogger("goalway.evaluation")


# ---------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------

def _ids_at_k(results: list, k: int) -> list[str]:
    return [result.chunk.document_id for result in results[:k]]


def recall_at_k(results: list, relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of relevant documents present in the top-k."""
    if not relevant_ids:
        return 0.0

    top_ids = set(_ids_at_k(results, k))
    return len(top_ids & relevant_ids) / len(relevant_ids)


def precision_at_k(results: list, relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of top-k results that are relevant."""
    if not results:
        return 0.0

    top_ids = set(_ids_at_k(results, k))
    return len(top_ids & relevant_ids) / len(top_ids)


def mrr(results: list, relevant_ids: set[str]) -> float:
    """Mean reciprocal rank of the first relevant result."""
    for rank, result in enumerate(results, start=1):
        if result.chunk.document_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(results: list, relevant_ids: set[str], k: int = 5) -> float:
    """
    Normalized discounted cumulative gain. Binary relevance:
    relevant documents count as 1.
    """
    dcg = 0.0

    for rank, result in enumerate(results[:k], start=1):
        if result.chunk.document_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    query: str,
    results: list,
    relevant_ids: list[str],
    k: int = 5,
) -> dict:
    """Full retrieval metric report for one query."""
    relevant = set(relevant_ids)

    return {
        "query": query,
        "k": k,
        "returned_results": len(results),
        "relevant_ids": sorted(relevant),
        "recall_at_k": round(recall_at_k(results, relevant, k), 4),
        "precision_at_k": round(precision_at_k(results, relevant, k), 4),
        "mrr": round(mrr(results, relevant), 4),
        "ndcg_at_k": round(ndcg_at_k(results, relevant, k), 4),
    }


# ---------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating an AI answer for groundedness and quality.

Question:
{query}

Retrieved context:
{context}

Answer to evaluate:
{answer}

Respond with ONLY a JSON object:
{{"score": <1-5 integer>, "grounded": <true|false>, "reasoning": "<one sentence>"}}

Scoring:
5 = fully supported by the context, complete and correct
4 = supported with minor omissions
3 = partially supported
2 = mostly unsupported
1 = contradicts the context or is fabricated
"""


def _answer_text(answer) -> str:
    if isinstance(answer, str):
        return answer

    # OpenAI-shaped response dict
    try:
        return answer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return str(answer)


def _context_text(results: list, max_chars: int = 4000) -> str:
    parts = []

    for result in results:
        parts.append(
            f"[{result.chunk.document_id}] {result.chunk.text}"
        )

    text = "\n\n".join(parts)
    return text[:max_chars]


def _parse_judge_response(text: str) -> dict | None:
    """
    Parse the judge's JSON. Models sometimes wrap JSON in code fences
    or add prose around it; scan for the first parseable JSON object
    rather than grabbing greedily from first { to last }.
    """
    import json

    decoder = json.JSONDecoder()

    stripped = text.strip()

    # Strip code fences first so the JSON starts at position 0.
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()

        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    for index, char in enumerate(stripped):
        if char != "{":
            continue

        try:
            data, _ = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict) and isinstance(
            data.get("score"), (int, float)
        ):
            return {
                "score": max(1, min(5, int(data["score"]))),
                "grounded": bool(data.get("grounded", False)),
                "reasoning": str(data.get("reasoning", ""))[:500],
            }

    return None


def llm_judge(
    query: str,
    answer,
    results: list,
    model: str = "auto",
) -> dict:
    """
    Judge an answer with an LLM; fall back to the deterministic
    groundedness report when no model is available.
    """
    answer_text = _answer_text(answer)
    context = _context_text(results)

    try:
        from gateway.llm import complete

        response = complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        query=query,
                        context=context,
                        answer=answer_text,
                    ),
                }
            ],
            temperature=0.0,
        )

        parsed = _parse_judge_response(
            response["choices"][0]["message"]["content"]
        )

        if parsed:
            parsed["judge"] = "llm"
            parsed["model"] = model
            return parsed

    except Exception as exc:
        logger.info("LLM judge unavailable (%s), using heuristic", exc)

    report = groundedness_report(answer_text, results)
    score = 1 + round(report["groundedness_score"] * 4)

    return {
        "score": score,
        "grounded": report["verdict"] == "grounded",
        "reasoning": report["summary"],
        "judge": "heuristic",
        "model": None,
    }


def evaluate_answer(
    query: str,
    answer,
    results: list,
    relevant_ids: list[str] | None = None,
    k: int = 5,
    model: str = "auto",
) -> dict:
    """
    Full evaluation report for one RAG turn: retrieval metrics
    (when ground truth is given) plus answer quality (LLM judge with
    heuristic fallback).
    """
    evaluation: dict = {
        "query": query,
    }

    if relevant_ids:
        evaluation["retrieval"] = evaluate_retrieval(
            query,
            results,
            relevant_ids,
            k=k,
        )

    evaluation["answer"] = llm_judge(query, answer, results, model=model)

    return evaluation
