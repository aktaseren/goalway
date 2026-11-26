from gateway.providers.granite import (
    DEFAULT_GRANITE_EMBEDDING_MODEL,
    DEFAULT_GRANITE_GENERATION_MODEL,
    load_granite_config,
)


def test_loads_default_granite_configuration(
    monkeypatch,
):
    variables = [
        "GRANITE_GENERATION_BACKEND",
        "GRANITE_GENERATION_MODEL",
        "GRANITE_GENERATION_BASE_URL",
        "GRANITE_EMBEDDING_BACKEND",
        "GRANITE_EMBEDDING_MODEL",
        "GRANITE_EMBEDDING_BASE_URL",
        "GRANITE_EMBEDDING_DIMENSIONS",
    ]

    for variable in variables:
        monkeypatch.delenv(
            variable,
            raising=False,
        )

    config = load_granite_config()

    assert config.generation.provider == (
        "openshift_ai"
    )
    assert config.generation.model == (
        DEFAULT_GRANITE_GENERATION_MODEL
    )

    assert config.embedding.provider == (
        "huggingface"
    )
    assert config.embedding.model == (
        DEFAULT_GRANITE_EMBEDDING_MODEL
    )
    assert config.embedding.dimensions == 384


def test_loads_remote_granite_embedding_config(
    monkeypatch,
):
    monkeypatch.setenv(
        "GRANITE_EMBEDDING_BACKEND",
        "vllm",
    )
    monkeypatch.setenv(
        "GRANITE_EMBEDDING_BASE_URL",
        "http://embedding:8000/v1",
    )
    monkeypatch.setenv(
        "GRANITE_EMBEDDING_MODEL",
        "granite-embedding",
    )
    monkeypatch.setenv(
        "GRANITE_EMBEDDING_DIMENSIONS",
        "768",
    )

    config = load_granite_config()

    assert config.embedding.provider == "vllm"
    assert config.embedding.base_url == (
        "http://embedding:8000/v1"
    )
    assert config.embedding.dimensions == 768