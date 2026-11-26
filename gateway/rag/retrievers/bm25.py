import math
import re
from collections import Counter

from gateway.rag.models import DocumentChunk
from gateway.rag.retrievers.base import Retriever
from gateway.rag.retrievers.models import (
    MetadataFilter,
    RetrievalResult,
)


_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever(Retriever):
    def __init__(
        self,
        chunks: list[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("BM25 k1 must be greater than zero")

        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")

        self._chunks = chunks
        self._k1 = k1
        self._b = b

        self._documents = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self._document_frequencies: Counter[str] = Counter()

        for document in self._documents:
            self._document_frequencies.update(
                set(document)
            )

        self._average_document_length = (
            sum(len(document) for document in self._documents)
            / len(self._documents)
            if self._documents
            else 0.0
        )

    @property
    def name(self) -> str:
        return "bm25"

    def _idf(self, term: str) -> float:
        document_count = len(self._documents)
        frequency = self._document_frequencies.get(term, 0)

        return math.log(
            1
            + (
                document_count
                - frequency
                + 0.5
            )
            / (
                frequency
                + 0.5
            )
        )

    def _score(
        self,
        query_terms: list[str],
        document: list[str],
    ) -> float:
        if not document:
            return 0.0

        frequencies = Counter(document)
        document_length = len(document)

        score = 0.0

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            denominator = (
                term_frequency
                + self._k1
                * (
                    1
                    - self._b
                    + self._b
                    * document_length
                    / max(
                        self._average_document_length,
                        1.0,
                    )
                )
            )

            score += self._idf(term) * (
                term_frequency
                * (
                    self._k1
                    + 1
                )
                / denominator
            )

        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_terms = tokenize(query)

        scored: list[
            tuple[DocumentChunk, float]
        ] = []

        for chunk, document in zip(
            self._chunks,
            self._documents,
        ):
            chunk_metadata = getattr(
                chunk,
                "metadata",
                {},
            ) or {}

            if (
                metadata_filter is not None
                and not metadata_filter.matches(
                    chunk_metadata
                )
            ):
                continue

            score = self._score(
                query_terms,
                document,
            )

            if score > 0:
                scored.append(
                    (
                        chunk,
                        score,
                    )
                )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=index,
                retriever=self.name,
            )
            for index, (chunk, score) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]