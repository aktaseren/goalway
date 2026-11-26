import pytest

from gateway.rag.strategy_selector import (
    StrategySelector,
    select_strategy,
)
from gateway.rag.retrievers import RetrievalStrategy


def test_selector_chooses_lexical_for_ids():
    selector = StrategySelector()

    assert selector.select("JIRA-123") == RetrievalStrategy.LEXICAL
    assert selector.select("Find document DOC-456") == RetrievalStrategy.LEXICAL
    assert selector.select("ticket id 12345") == RetrievalStrategy.LEXICAL


def test_selector_chooses_lexical_for_exact_phrases():
    selector = StrategySelector()

    assert selector.select('"exact phrase match"') == RetrievalStrategy.LEXICAL
    assert selector.select('code "openshift-ai"') == RetrievalStrategy.LEXICAL


def test_selector_chooses_lexical_for_exact_keywords():
    selector = StrategySelector()

    assert selector.select("find the exact code") == RetrievalStrategy.LEXICAL
    assert selector.select("get uuid for document") == RetrievalStrategy.LEXICAL
    assert selector.select("number for this item") == RetrievalStrategy.LEXICAL


def test_selector_chooses_vector_for_semantic_queries():
    selector = StrategySelector()

    assert selector.select("What is OpenShift?") == RetrievalStrategy.VECTOR
    assert selector.select("How does model serving work?") == RetrievalStrategy.VECTOR
    assert selector.select("Why use Kubernetes?") == RetrievalStrategy.VECTOR
    assert selector.select("Explain embeddings") == RetrievalStrategy.VECTOR
    assert selector.select("Similar to vector search") == RetrievalStrategy.VECTOR


def test_selector_chooses_vector_when_starting_with_semantic_word():
    selector = StrategySelector()

    assert selector.select("what about this") == RetrievalStrategy.VECTOR
    assert selector.select("how many results") == RetrievalStrategy.VECTOR
    assert selector.select("explain the process") == RetrievalStrategy.VECTOR


def test_selector_chooses_hybrid_for_mixed_queries():
    selector = StrategySelector()

    assert selector.select("OpenShift model serving") == RetrievalStrategy.HYBRID
    assert selector.select("deployment strategies") == RetrievalStrategy.HYBRID
    assert selector.select("best practices for AI") == RetrievalStrategy.HYBRID


def test_selector_with_custom_rules():
    rules = {
        "exact_match_keywords": ["ticket", "issue"],
        "semantic_indicators": ["recommend", "suggest"],
    }

    selector = StrategySelector(rules)

    assert selector.select("find ticket") == RetrievalStrategy.LEXICAL
    assert selector.select("recommend a solution") == RetrievalStrategy.VECTOR


def test_select_strategy_respects_explicit_request():
    result = select_strategy(
        query="What is OpenShift?",
        requested_strategy=RetrievalStrategy.LEXICAL,
        auto_select=True,
    )

    # Explicit request overrides auto-selection
    assert result == RetrievalStrategy.LEXICAL


def test_select_strategy_auto_selects_when_enabled():
    result = select_strategy(
        query="What is OpenShift?",
        requested_strategy=None,
        auto_select=True,
    )

    # Should auto-select VECTOR for semantic query
    assert result == RetrievalStrategy.VECTOR


def test_select_strategy_defaults_to_hybrid_when_disabled():
    result = select_strategy(
        query="What is OpenShift?",
        requested_strategy=None,
        auto_select=False,
    )

    # Should default to HYBRID when auto-select is off
    assert result == RetrievalStrategy.HYBRID


def test_select_strategy_with_custom_rules():
    rules = {
        "exact_match_keywords": ["ticket"],
        "semantic_indicators": ["recommend"],
    }

    result = select_strategy(
        query="find ticket ABC",
        requested_strategy=None,
        auto_select=True,
        rules=rules,
    )

    assert result == RetrievalStrategy.LEXICAL
