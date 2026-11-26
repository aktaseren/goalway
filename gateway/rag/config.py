import os
import json
from pathlib import Path
from gateway.rag.retrievers import RetrievalStrategy


class RAGConfig:
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or os.getenv(
            "RAG_CONFIG_FILE",
            "config/rag_config.json"
        )
        self._config = self._load_config()

    def _load_config(self) -> dict:
        config_file = Path(self.config_path)

        if not config_file.exists():
            return self._default_config()

        try:
            with open(config_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return self._default_config()

    def _default_config(self) -> dict:
        return {
            "retrieval": {
                "default_strategy": "hybrid",
                "strategies": {
                    "hybrid": {
                        "enabled": True,
                        "candidate_multiplier": 4,
                        "rank_constant": 60,
                    }
                },
                "auto_strategy_selection": {
                    "enabled": False,
                }
            },
            "filtering": {
                "metadata": {
                    "enabled": True,
                }
            },
            "statistics": {
                "include_by_default": False,
            }
        }

    @property
    def default_strategy(self) -> RetrievalStrategy:
        strategy_str = os.getenv(
            "RAG_DEFAULT_STRATEGY",
            self._config["retrieval"]["default_strategy"]
        )
        return RetrievalStrategy(strategy_str)

    @property
    def auto_strategy_enabled(self) -> bool:
        return self._config["retrieval"]["auto_strategy_selection"]["enabled"]

    @property
    def auto_strategy_rules(self) -> dict:
        return self._config["retrieval"]["auto_strategy_selection"].get("rules", {})

    @property
    def hybrid_candidate_multiplier(self) -> int:
        return self._config["retrieval"]["strategies"]["hybrid"].get(
            "candidate_multiplier",
            4
        )

    @property
    def hybrid_rank_constant(self) -> int:
        return self._config["retrieval"]["strategies"]["hybrid"].get(
            "rank_constant",
            60
        )

    @property
    def include_stats_by_default(self) -> bool:
        return self._config["statistics"]["include_by_default"]


_rag_config: RAGConfig | None = None


def get_rag_config() -> RAGConfig:
    global _rag_config
    if _rag_config is None:
        _rag_config = RAGConfig()
    return _rag_config
