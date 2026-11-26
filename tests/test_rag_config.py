import os
import json
import tempfile
import pytest

from gateway.rag.config import RAGConfig
from gateway.rag.retrievers import RetrievalStrategy


def test_config_loads_default_strategy():
    config = RAGConfig()
    assert config.default_strategy == RetrievalStrategy.HYBRID


def test_config_loads_from_file():
    test_config = {
        "retrieval": {
            "default_strategy": "vector",
            "strategies": {
                "hybrid": {
                    "enabled": True,
                    "candidate_multiplier": 3,
                    "rank_constant": 50,
                }
            },
            "auto_strategy_selection": {
                "enabled": True,
            }
        },
        "statistics": {
            "include_by_default": True,
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(test_config, f)
        config_path = f.name

    try:
        config = RAGConfig(config_path)

        assert config.default_strategy == RetrievalStrategy.VECTOR
        assert config.auto_strategy_enabled is True
        assert config.hybrid_candidate_multiplier == 3
        assert config.hybrid_rank_constant == 50
        assert config.include_stats_by_default is True
    finally:
        os.unlink(config_path)


def test_config_environment_variable_overrides_file():
    test_config = {
        "retrieval": {
            "default_strategy": "vector",
            "strategies": {
                "hybrid": {
                    "enabled": True,
                }
            },
            "auto_strategy_selection": {
                "enabled": False,
            }
        },
        "statistics": {
            "include_by_default": False,
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(test_config, f)
        config_path = f.name

    try:
        os.environ["RAG_DEFAULT_STRATEGY"] = "lexical"
        config = RAGConfig(config_path)

        assert config.default_strategy == RetrievalStrategy.LEXICAL
    finally:
        os.unlink(config_path)
        del os.environ["RAG_DEFAULT_STRATEGY"]


def test_config_uses_defaults_when_file_missing():
    config = RAGConfig("/nonexistent/path/config.json")

    assert config.default_strategy == RetrievalStrategy.HYBRID
    assert config.auto_strategy_enabled is False
    assert config.hybrid_candidate_multiplier == 4
    assert config.hybrid_rank_constant == 60


def test_config_handles_malformed_json():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write("{ invalid json }")
        config_path = f.name

    try:
        config = RAGConfig(config_path)
        # Should fall back to defaults without crashing
        assert config.default_strategy == RetrievalStrategy.HYBRID
    finally:
        os.unlink(config_path)
