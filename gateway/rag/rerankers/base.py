class Reranker:
    """
    Reranks retrieval candidates against the query with a finer-grained
    relevance model than first-stage retrieval .
    """

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5,
    ) -> list:
        """
        Takes a list of SearchResult-like objects (must expose .chunk
        and .score) and returns them reordered by relevance.
        """
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError
