from gateway.rag.chunker import TextChunker
from gateway.rag.embeddings import (
    EmbeddingProvider,
    create_embedding_provider,
)
from gateway.rag.models import Document, SearchResult
from gateway.rag.vector_store import VectorStore
from gateway.rag.retrievers import (
    BM25Retriever,
    VectorRetriever,
    HybridRetriever,
    MetadataFilter,
    RetrievalStrategy,
    RetrievalStats,
)
from gateway.rag.config import get_rag_config

from gateway.rag.sqlite_vector_store import SQLiteVectorStore

class RAGService:
    def __init__(
        self,
        chunker: TextChunker | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.chunker = chunker or TextChunker()
        self.embedding_provider = (
            embedding_provider or create_embedding_provider()
        )
        self.vector_store = vector_store or SQLiteVectorStore()

        # Lexical index: preload every stored chunk so restarts keep
        # the full knowledge base (BM25 runs over this cache).
        self._chunks_cache: list = self._load_chunks()

    def _load_chunks(self) -> list:
        loader = getattr(self.vector_store, "all_chunks", None)

        if loader is None:
            return []

        try:
            return loader()
        except Exception:
            return []

    def ingest(self, document: Document) -> int:
        chunks = self.chunker.chunk(document)

        embeddings = self.embedding_provider.embed(
            [chunk.text for chunk in chunks]
        )

        self.vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        # Replace-by-document in the in-memory lexical cache too:
        # re-ingesting must not duplicate chunks.
        self._chunks_cache = [
            chunk
            for chunk in self._chunks_cache
            if chunk.document_id != document.document_id
        ]
        self._chunks_cache.extend(chunks)

        return len(chunks)

    def get_document(self, document_id: str) -> list:
        """
        All chunks of one document in position order. Returns []
        when the store does not support document fetches.
        """
        fetcher = getattr(self.vector_store, "get_document_chunks", None)

        if fetcher is None:
            return []

        try:
            return fetcher(document_id)
        except Exception:
            return []

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[list[SearchResult], RetrievalStats | None]:
        if strategy == RetrievalStrategy.LEXICAL:
            retriever = BM25Retriever(self._chunks_cache)
            results = retriever.search(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            stats = RetrievalStats(
                strategy=RetrievalStrategy.LEXICAL,
                query=query,
                returned_results=len(results),
                lexical_candidates=len(results),
            )

        elif strategy == RetrievalStrategy.VECTOR:
            retriever = VectorRetriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
            )
            results = retriever.search(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            stats = RetrievalStats(
                strategy=RetrievalStrategy.VECTOR,
                query=query,
                returned_results=len(results),
                vector_candidates=len(results),
            )

        elif strategy == RetrievalStrategy.HYBRID:
            rag_config = get_rag_config()
            lexical = BM25Retriever(self._chunks_cache)
            vector = VectorRetriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
            )
            retriever = HybridRetriever(
                lexical_retriever=lexical,
                vector_retriever=vector,
                candidate_multiplier=rag_config.hybrid_candidate_multiplier,
                rank_constant=rag_config.hybrid_rank_constant,
            )
            results = retriever.search(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            stats = retriever.last_stats

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        search_results = [
            SearchResult(
                chunk=result.chunk,
                score=result.score,
            )
            for result in results
        ]

        return search_results, stats