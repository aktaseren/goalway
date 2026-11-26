import pytest

from gateway.rag.embeddings.config import (
    EmbeddingConfig,
    load_embedding_config,
)


def test_loads_default_dummy_configuration(monkeypatch):
    monkeypatch.delenv(
        "EMBEDDING_PROVIDER",
        raising=False,
    )
    monkeypatch.delenv(
        "EMBEDDING_MODEL",
        raising=False,
    )

    config = load_embedding_config()

    assert config.provider == "dummy"
    assert config.model == "dummy"
    assert config.timeout_seconds == 120
    assert config.batch_size == 32


def test_loads_ollama_configuration(monkeypatch):
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "EMBEDDING_MODEL",
        "embeddinggemma",
    )
    monkeypatch.setenv(
        "EMBEDDING_BASE_URL",
        "http://ollama-test:11434",
    )
    monkeypatch.setenv(
        "EMBEDDING_TIMEOUT_SECONDS",
        "60",
    )
    monkeypatch.setenv(
        "EMBEDDING_DIMENSIONS",
        "768",
    )
    monkeypatch.setenv(
        "EMBEDDING_BATCH_SIZE",
        "16",
    )

    config = load_embedding_config()

    assert config.provider == "ollama"
    assert config.model == "embeddinggemma"
    assert config.base_url == "http://ollama-test:11434"
    assert config.timeout_seconds == 60
    assert config.dimensions == 768
    assert config.batch_size == 16


def test_rejects_invalid_timeout():
    with pytest.raises(
        ValueError,
        match="timeout",
    ):
        EmbeddingConfig(
            provider="ollama",
            model="embeddinggemma",
            timeout_seconds=0,
        )


def test_rejects_invalid_batch_size():
    with pytest.raises(
        ValueError,
        match="batch size",
    ):
        EmbeddingConfig(
            provider="ollama",
            model="embeddinggemma",
            batch_size=0,
        )