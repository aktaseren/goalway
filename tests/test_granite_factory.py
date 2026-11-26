from unittest.mock import patch

from gateway.providers.common import ProviderConfig
from gateway.providers.generation import (
    OpenAICompatibleGenerationProvider,
)
from gateway.providers.granite import (
    GraniteConfig,
    create_granite_embedding_provider,
    create_granite_generation_provider,
)
from gateway.rag.embeddings import (
    HuggingFaceEmbeddingProvider,
)
from gateway.rag.embeddings.config import (
    EmbeddingConfig,
)


def granite_config() -> GraniteConfig:
    return GraniteConfig(
        generation=ProviderConfig(
            provider="openshift_ai",
            model="ibm-granite/granite-4.1-3b",
            base_url="http://granite:8000/v1",
            batch_size=1,
        ),
        embedding=EmbeddingConfig(
            provider="huggingface",
            model=(
                "ibm-granite/"
                "granite-embedding-small-english-r2"
            ),
            dimensions=384,
        ),
        embedding_backend="huggingface",
    )


def test_creates_granite_generation_provider():
    provider = create_granite_generation_provider(
        granite_config()
    )

    assert isinstance(
        provider,
        OpenAICompatibleGenerationProvider,
    )
    assert provider.config.provider == (
        "openshift_ai"
    )
    assert provider.config.model == (
        "ibm-granite/granite-4.1-3b"
    )


@patch(
    "gateway.rag.embeddings.huggingface."
    "SentenceTransformer"
)
def test_creates_granite_embedding_provider(
    mock_transformer,
):
    provider = create_granite_embedding_provider(
        granite_config()
    )

    assert isinstance(
        provider,
        HuggingFaceEmbeddingProvider,
    )
    assert provider.config.dimensions == 384

    mock_transformer.assert_called_once_with(
        "ibm-granite/"
        "granite-embedding-small-english-r2"
    )