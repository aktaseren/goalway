from abc import ABC, abstractmethod

from gateway.rag.retrievers.models import (
    MetadataFilter,
    RetrievalResult,
)


class Retriever(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError