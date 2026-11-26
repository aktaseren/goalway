import os

import pytest

from gateway.rag.embeddings import (
    DummyEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from gateway.rag.embeddings.factory import create_embedding_provider


def test_factory_creates_dummy_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "dummy")

    provider = create_embedding_provider()

    assert isinstance(provider, DummyEmbeddingProvider)


def test_factory_creates_ollama_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-test:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "45")

    provider = create_embedding_provider()

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.base_url == "http://ollama-test:11434"
    assert provider.model == "embeddinggemma"
    assert provider.timeout_seconds == 45


@pytest.mark.parametrize(
    "provider_name",
    [
        "openai",
        "openai_compatible",
        "vllm",
    ],
)
def test_factory_creates_openai_compatible_provider(
    monkeypatch,
    provider_name,
):
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider_name)
    monkeypatch.setenv(
        "EMBEDDING_BASE_URL",
        "http://embedding-service:8000/v1",
    )
    monkeypatch.setenv(
        "EMBEDDING_MODEL",
        "test-embedding-model",
    )
    monkeypatch.setenv(
        "EMBEDDING_API_KEY",
        "test-secret",
    )
    monkeypatch.setenv(
        "EMBEDDING_TIMEOUT_SECONDS",
        "75",
    )

    provider = create_embedding_provider()

    assert isinstance(
        provider,
        OpenAICompatibleEmbeddingProvider,
    )
    assert provider.base_url == "http://embedding-service:8000/v1"
    assert provider.model == "test-embedding-model"
    assert provider.api_key == "test-secret"
    assert provider.timeout_seconds == 75


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER",
        "unknown-provider",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported embedding provider",
    ):
        create_embedding_provider()