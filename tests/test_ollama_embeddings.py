import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from unittest.mock import Mock, patch

from gateway.rag.embeddings import OllamaEmbeddingProvider

from gateway.rag.embeddings.config import EmbeddingConfig

@patch("gateway.rag.embeddings.ollama.requests.post")
def test_ollama_embedding_provider(mock_post):
    response = Mock()
    response.json.return_value = {
        "embeddings": [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    }
    response.raise_for_status.return_value = None
    mock_post.return_value = response

    config = EmbeddingConfig(
        provider="ollama",
        model="embeddinggemma",
        base_url="http://ollama:11434",
        timeout_seconds=120,
    )
    provider = OllamaEmbeddingProvider(config)

    embeddings = provider.embed(["first", "second"])

    assert embeddings == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    mock_post.assert_called_once_with(
        "http://ollama:11434/api/embed",
        json={
            "model": "embeddinggemma",
            "input": ["first", "second"],
        },
        timeout=120,
    )