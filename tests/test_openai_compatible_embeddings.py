from unittest.mock import Mock, patch

import pytest

from gateway.rag.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)

from gateway.rag.embeddings.config import EmbeddingConfig

@patch(
    "gateway.rag.embeddings.openai_compatible.requests.post"
)
def test_openai_compatible_provider_returns_vectors(mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {
                "index": 1,
                "embedding": [0.3, 0.4],
            },
            {
                "index": 0,
                "embedding": [0.1, 0.2],
            },
        ]
    }
    mock_post.return_value = response

    config = EmbeddingConfig(
        provider = "openai",
        model="text-embedding-3-small",
        base_url="http://embedding-service:8000/v1",
        api_key="secret",
    )

    provider = OpenAICompatibleEmbeddingProvider(config)

    result = provider.embed(
        [
            "first text",
            "second text",
        ]
    )

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    mock_post.assert_called_once_with(
        "http://embedding-service:8000/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer secret",
        },
        json={
            "model": "test-embedding-model",
            "input": [
                "first text",
                "second text",
            ],
        },
        timeout=120,
    )


@patch(
    "gateway.rag.embeddings.openai_compatible.requests.post"
)
def test_openai_compatible_provider_works_without_api_key(
    mock_post,
):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {
                "index": 0,
                "embedding": [0.5, 0.6],
            }
        ]
    }
    mock_post.return_value = response

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://vllm:8000/v1/",
        model="embedding-model",
    )

    result = provider.embed(["hello"])

    assert result == [[0.5, 0.6]]

    called_headers = mock_post.call_args.kwargs["headers"]

    assert "Authorization" not in called_headers


def test_openai_compatible_provider_handles_empty_input():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://vllm:8000/v1",
        model="embedding-model",
    )

    assert provider.embed([]) == []


@patch(
    "gateway.rag.embeddings.openai_compatible.requests.post"
)
def test_openai_compatible_provider_rejects_invalid_response(
    mock_post,
):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "unexpected": [],
    }
    mock_post.return_value = response

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://vllm:8000/v1",
        model="embedding-model",
    )

    with pytest.raises(
        ValueError,
        match="did not contain a data list",
    ):
        provider.embed(["hello"])