from unittest.mock import MagicMock, patch

from gateway.rag.embeddings.huggingface import (
    HuggingFaceEmbeddingProvider,
)

from gateway.rag.embeddings.config import EmbeddingConfig

@patch(
    "gateway.rag.embeddings.huggingface.SentenceTransformer"
)
def test_huggingface_provider(mock_model):

    fake = MagicMock()

    fake.encode.return_value = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]

    mock_model.return_value =fake

    config = EmbeddingConfig(
        provider="huggingface",
        model="test-model",
    )

    provider = HuggingFaceEmbeddingProvider(config)

    result = provider.embed(
        [
            "hello",
            "world",
        ]
    )

    assert result == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]