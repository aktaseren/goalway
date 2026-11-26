import re
from gateway.rag.retrievers import RetrievalStrategy


class StrategySelector:
    """
    Selects optimal retrieval strategy based on query characteristics.

    Heuristics:
    - LEXICAL: Queries with exact identifiers, codes, numbers
    - VECTOR: Semantic queries (how, why, explain, similar to)
    - HYBRID: Default for mixed or unclear queries
    """

    EXACT_MATCH_PATTERNS = [
        r'\b[A-Z]{2,}-\d+\b',          # IDs like "JIRA-123", "DOC-456"
        r'\b[0-9a-f]{8}-[0-9a-f]{4}\b', # UUIDs (partial)
        r'\b\d{5,}\b',                  # Long numbers
        r'"[^"]+"',                     # Quoted exact phrases
    ]

    EXACT_MATCH_KEYWORDS = {
        "id", "code", "number", "uuid", "key", "identifier",
        "exact", "specifically", "precisely"
    }

    SEMANTIC_KEYWORDS = {
        "what", "how", "why", "explain", "describe", "understand",
        "similar", "like", "related", "about", "concept", "meaning",
        "difference", "compare", "overview", "summary"
    }

    def __init__(self, rules: dict | None = None):
        self.rules = rules or {}
        self.exact_keywords = set(
            self.rules.get("exact_match_keywords", [])
        ) | self.EXACT_MATCH_KEYWORDS

        self.semantic_keywords = set(
            self.rules.get("semantic_indicators", [])
        ) | self.SEMANTIC_KEYWORDS

    def select(self, query: str) -> RetrievalStrategy:
        """
        Select optimal retrieval strategy for the given query.

        Args:
            query: User query string

        Returns:
            RetrievalStrategy enum value
        """
        query_lower = query.lower()

        # Check for exact match patterns
        for pattern in self.EXACT_MATCH_PATTERNS:
            if re.search(pattern, query):
                return RetrievalStrategy.LEXICAL

        # Check for exact match keywords
        query_words = set(query_lower.split())
        if query_words & self.exact_keywords:
            return RetrievalStrategy.LEXICAL

        # Check for semantic indicators
        if query_words & self.semantic_keywords:
            return RetrievalStrategy.VECTOR

        # Check if query starts with semantic word
        first_word = query_lower.split()[0] if query_lower else ""
        if first_word in self.semantic_keywords:
            return RetrievalStrategy.VECTOR

        # Default to hybrid for mixed/unclear queries
        return RetrievalStrategy.HYBRID


def select_strategy(
    query: str,
    requested_strategy: RetrievalStrategy | None = None,
    auto_select: bool = False,
    rules: dict | None = None,
) -> RetrievalStrategy:
    """
    Select retrieval strategy based on request and query characteristics.

    Priority:
    1. Explicitly requested strategy (highest priority)
    2. Auto-selection based on query (if enabled)
    3. Default strategy (fallback)

    Args:
        query: User query
        requested_strategy: Strategy explicitly requested by user
        auto_select: Whether to enable automatic strategy selection
        rules: Custom rules for auto-selection

    Returns:
        Selected RetrievalStrategy
    """
    # Priority 1: Explicit request
    if requested_strategy is not None:
        return requested_strategy

    # Priority 2: Auto-selection
    if auto_select:
        selector = StrategySelector(rules)
        return selector.select(query)

    # Priority 3: Default (hybrid)
    return RetrievalStrategy.HYBRID
